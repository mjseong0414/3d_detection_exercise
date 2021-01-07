import subprocess # 다른 언어로 만들어진 프로그램을 통합, 제어할 수 있게 만드는 모듈
from pathlib import Path

from .version import __version__

__all__=[
    '__version__'
]


def get_git_commit_number():
    if not (Path(__file__).parent / '../.git').exists: # 3D_DETECTION폴더에 .git이 없으면 0000000을 리턴
        return '0000000'
    
    cmd_out = subprocess.run(['git', 'rev-parse', 'HEAD'], stdout=subprocess.PIPE)
    git_commit_number = cmd_out.stdout.decode('utf-8')[:7]
    return git_commit_number


script_version = get_git_commit_number()


if script_version not in __version__:
    __version__ = __version__ + '+py%s' % script_version
    