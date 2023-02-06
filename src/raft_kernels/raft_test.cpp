class hi : public raft::kernel
{
  public:
    hi()
      : raft::kernel()
    {
        output.addPort<std::string>("0");
    }

    virtual raft::kstatus run()
    {

        output["0"].push(std::string("Hello World\n"));
        return (raft::stop);
    }
};

class simple
{
  private:
    std::shared_ptr<int> test{ NULL };
    int _ntests{ 0 };
    int garbage{ 0 };

  public:
    simple(int ntests = 2)
    {
        std::cout << "allocating " << ntests << " bytes\n";
        _ntests = ntests;
        test.reset((int*)(malloc(sizeof(int) * ntests)), free);
        if (test == NULL) {
            _ntests = 0;
        }
        test.get()[0] = 999;
    }

    int get_size()
    {
        return _ntests;
    }

    simple(const simple& t)
    {
        std::cout << "Copying " << _ntests << " bytes\n";
        test = t.test;
        _ntests = t._ntests;
        garbage = t.garbage;
        test = t.test;
    }

    ~simple()
    {
        if (_ntests > 0) {
            std::cout << "Freeing memory " << _ntests << " bytes " << test.get()[0] << "\n";
            std::cout << "Count: " << test.use_count() << std::endl;
            // test.reset();
            std::cout << "Count: " << test.use_count() << std::endl;
            if (test.unique()) {
                // delete test;
                std::cout << "Cleaning " << _ntests << " bytes\n";
                test.reset();
            }
            _ntests = 0;
        } else {
            std::cout << "Nothing to free!\n";
        }
    }

    void increase_garbage()
    {
        ++garbage;
    }

    int get_garbage()
    {
        return garbage;
    }
};

class simple_producer : public raft::kernel
{
  public:
    simple_producer()
      : raft::kernel()
    {
        output.addPort<simple>("1");
    }

    virtual raft::kstatus run()
    {
        for (int i = 1; i < 4; ++i) {
            auto& out_simple(output["1"].template allocate<simple>(i));
            output["1"].send();
        }

        return (raft::stop);
    }
};

class simple_consumer : public raft::kernel
{
  public:
    simple_consumer()
      : raft::kernel()
    {
        input.addPort<simple>("0");
        output.addPort<simple>("1");
    }

    virtual raft::kstatus run()
    {

        auto& peek(input["0"].template peek<simple>());
        peek.increase_garbage();
        std::cout << "Consuming " << peek.get_size() << " bytes \n";
        std::cout << "Size FIFO: " << input["0"].size() << "\n";

        output["1"].push(peek);
        input["0"].unpeek();
        input["0"].recycle(1);

        return (raft::proceed);
    }
};

class simple_post : public raft::kernel
{
  public:
    simple_post()
      : raft::kernel()
    {
        input.addPort<simple>("0");
        output.addPort<int>("1");
    }

    virtual raft::kstatus run()
    {
        // simple peek;
        // input["0"].pop(peek);
        auto& peek(input["0"].template peek<simple>());
        peek.increase_garbage();
        output["1"].push(peek.get_garbage());
        std::cout << "Getting garbage " << peek.get_garbage() << "\n";
        input["0"].recycle(1);

        return (raft::proceed);
    }
};

// enum opt{
// #if defined __APPLE__ && __APPLE__
//     DEFAULT_SOCK_BUF_SIZE  = 4*1024*1024,
// 		DEFAULT_LINGER_SECS    = 1,
// #else
// 		DEFAULT_SOCK_BUF_SIZE  = 256*1024*1024,
// 		DEFAULT_LINGER_SECS    = 3,
// #endif
// 		DEFAULT_MAX_CONN_QUEUE = 128
// 	} opt;

struct __attribute__((packed)) chips_hdr_type1
{
    uint8_t roach;    // Note: 1-based
    uint8_t gbe;      // (AKA tuning)
    uint8_t nchan;    // 109
    uint8_t nsubband; // 11
    uint8_t subband;  // 0-11
    uint8_t nroach;   // 16
    // Note: Big endian
    uint16_t chan0; // First chan in packet
    uint64_t seq;   // Note: 1-based
};

struct __attribute__((packed)) chips_packet
{
    chips_hdr_type1 hdr;
    hwy::AlignedFreeUniquePtr<uint8_t[]> data{ nullptr };
    // chips_packet(int buf_size){
    //     data = std::move(hwy::AllocateAligned<uint8_t>(buf_size));
    // };
    // chips_packet(){};
};

struct __attribute__((packed)) num
{
    signed char real : 4;
    signed char img : 4;
};