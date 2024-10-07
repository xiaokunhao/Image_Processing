import os
def create_directory(dir: str = '') -> bool:
    if os.path.exists(dir):
        return True
    else:
        try:
            os.makedirs(dir)
            return False
        except OSError:
            return False
