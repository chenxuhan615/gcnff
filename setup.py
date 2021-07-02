import os
from setuptools import setup

def _process_requirements():
    packages = open('requirements.txt').read().strip().split('\n')
    requires = []
    for pkg in packages:
        if pkg.startswith('git+ssh'):
            return_code = os.system('pip3 install {}'.format(pkg))
            assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
        else:
            requires.append(pkg)

    # install pytorch
    return_code = os.system('pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html')
    assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
    # install torch-scatter
    return_code = os.system('pip3 install torch-scatter -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html')
    assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
    # install torch-sparse
    return_code = os.system('pip3 install torch-sparse -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html')
    assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
    # install torch-cluster
    return_code = os.system('pip3 install torch-cluster -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html')
    assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
    # install torch-spline-conv
    return_code = os.system('pip3 install torch-spline-conv -f https://pytorch-geometric.com/whl/torch-1.8.0+cu111.html')
    assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)
    # install torch-geometric
    return_code = os.system('pip3 install torch-geometric')
    assert return_code == 0, 'error, status_code is: {}, exit!'.format(return_code)

    return requires

setup(
    name="gcnff",
    version="1.0",
    author="hanchenxu",
    author_email="643814700@qq.com",
    description=("This is a molecular dynamics force field based on Graph Convolutional Networks."),
    license="xjtu",
    keywords="gcnff",
    packages=['gcnff'],

    # 需要安装的依赖
    install_requires=_process_requirements(),
    entry_points={'console_scripts': [
        'gcnff = gcnff.gcnff:main',
    ]},
    zip_safe=False
)
