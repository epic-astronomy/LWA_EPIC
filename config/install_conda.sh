set -xe

conda config --set always_yes yes --set changeps1 no
conda update -q conda
conda info -a
conda create --name=${ENV_NAME}  python=$PYTHON --quiet
conda env update -n ${ENV_NAME} -f config/${ENV_NAME}.yml
source activate ${ENV_NAME}


conda list -n ${ENV_NAME}
