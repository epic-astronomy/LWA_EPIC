 // struct ibv_device** device_list;
    int num_devices;
    auto* device_list = ibv_get_device_list(&num_devices);
    if (!device_list)
        exit(1);

    for (auto i = 0; i < num_devices; ++i) {
        std::cout << ibv_get_device_name(device_list[i]) << std::endl;
    }

    struct ibv_context* ctx;
    ctx = ibv_open_device(device_list[0]);
    if (!ctx) {
        fprintf(stderr, "Error, failed to open the device '% s'\n", ibv_get_device_name(device_list[0]));
        return -1;
    }
    printf("The device '% s' was opened\n", ibv_get_device_name(ctx->device));
    struct ibv_device_attr device_attr;
    int rc;
    rc = ibv_query_device(ctx, &device_attr);
    if (rc) {
        fprintf(stderr, "Error, failed to query the device '% s' attributes\n", ibv_get_device_name(device_list[0]));
        return -1;
    }
    struct ibv_pd* pd;
    pd = ibv_alloc_pd(ctx);
    if (!pd) {
        fprintf(stderr, "Error, ibv_alloc_pd() failed\n");
        return -1;
    }

    struct ibv_cq* cq;
    cq = ibv_create_cq(ctx, 100, NULL, NULL, 0);
    if (!cq) {
        fprintf(stderr, "Error, ibv_create_cq() failed\n");
        return -1;
    }

    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.send_cq = cq;
    qp_init_attr.recv_cq = cq;
    qp_init_attr.qp_type = IBV_QPT_UD;
    qp_init_attr.cap.max_send_wr = 2;
    qp_init_attr.cap.max_recv_wr = 2;
    qp_init_attr.cap.max_send_sge = 1;
    qp_init_attr.cap.max_recv_sge = 1;
    // qp_init_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;
    struct ibv_qp* qp = ibv_create_qp(pd, &qp_init_attr);
    if (!qp) {
        fprintf(stderr, "Error, ibv_create_qp() failed\n");
        return -1;
    }

    struct ibv_qp_attr init_attr;
    memset(&init_attr, 0, sizeof(init_attr));
    init_attr.qp_state = ibv_qp_state::IBV_QPS_INIT;
    init_attr.port_num = 1;
    init_attr.pkey_index = 0;
    init_attr.qkey = 0x80000000;
    init_attr.qp_access_flags = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE;

    ibv_modify_qp(qp, &init_attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);

    int ib_port = 1;
    ibv_port_attr port_attr;
    ibv_query_port(ctx, ib_port, &port_attr);
    struct ibv_qp_attr rtr_attr;
    memset(&rtr_attr, 0, sizeof(rtr_attr));
    rtr_attr.qp_state = ibv_qp_state::IBV_QPS_RTR;
    rtr_attr.path_mtu = ibv_mtu::IBV_MTU_4096;
    rtr_attr.rq_psn = 0;
    rtr_attr.max_dest_rd_atomic = 1;
    rtr_attr.min_rnr_timer = 0x12;
    rtr_attr.ah_attr.is_global = 0;
    rtr_attr.ah_attr.sl = 0;
    rtr_attr.ah_attr.src_path_bits = 0;
    rtr_attr.ah_attr.port_num = ib_port;

    rtr_attr.dest_qp_num = qp->qp_num;
    rtr_attr.ah_attr.dlid = port_attr.lid;
    ibv_modify_qp(qp, &rtr_attr, IBV_QP_STATE);

    // struct ibv_qp_attr rts_attr;
    // memset(&rts_attr, 0, sizeof(rts_attr));
    // rts_attr.qp_state = ibv_qp_state::IBV_QPS_RTS;
    // rts_attr.timeout = 0x12;
    // rts_attr.retry_cnt = 7;
    // rts_attr.rnr_retry = 7;
    // rts_attr.sq_psn = 0;
    // rts_attr.max_rd_atomic = 1;

    // ibv_modify_qp(qp, &init_attr, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);

    struct ibv_mr* mr;
    int buffer_size = 9000;
    auto buffer2 = hwy::AllocateAligned<uint8_t>(buffer_size);
    mr = ibv_reg_mr(pd, buffer2.get(), buffer_size, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ);
    if (!mr) {
        fprintf(stderr, "Error, ibv_reg_mr() failed\n");
        return -1;
    }

    struct ibv_recv_wr receive_wr, *bad_wr = nullptr;
    memset(&receive_wr, 0, sizeof(receive_wr));

    // RDMA supports scatter-gather I/O.
    // For a RECV operation, it works as scatter; received data will be scattered into several registered MR.
    struct ibv_sge* receive_sge = (ibv_sge*)calloc(sizeof(struct ibv_sge), 1);
    // for (int i = 0; i < sge.size(); i++) {
    receive_sge[0].addr = (uintptr_t)mr->addr;
    receive_sge[0].length = mr->length;
    receive_sge[0].lkey = mr->lkey;
    // }

    receive_wr.sg_list = receive_sge;
    receive_wr.num_sge = 1; // sge.size();
    // will be used for identification.
    // When a request fail, ibv_poll_cq() returns a work completion (struct ibv_wc) with the specified wr_id.
    // If the wr_id is 100, we can easily find out that this RECV request failed.
    receive_wr.wr_id = 100;
    // You can chain several receive requests to reduce software footprint, hnece to improve latency.
    receive_wr.next = nullptr;

    // If posting fails, the address of the failed WR among the chained WRs is stored in bad_wr.
    auto result = ibv_post_recv(qp, &receive_wr, &bad_wr);
    free(receive_sge);

    if (result != 0) {
        std::cout << "Receive request failed\n";
    }

    struct ibv_wc wc;
    std::cout<<"Polling\n";
    do {
        // ibv_poll_cq returns the number of WCs that are newly completed,
        // If it is 0, it means no new work completion is received.
        // Here, the second argument specifies how many WCs the poll should check,
        // however, giving more than 1 incurs stack smashing detection with g++8 compilation.
        result = ibv_poll_cq(cq, 1, &wc);
    } while (result == 0);

    if (result > 0 && wc.status == ibv_wc_status::IBV_WC_SUCCESS) {
        // success
        return true;
    }

    // You can identify which WR failed with wc.wr_id.
    printf("Poll failed with status %s (work request ID: %llu)\n", ibv_wc_status_str(wc.status), wc.wr_id);

    return false;