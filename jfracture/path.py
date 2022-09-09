from os.path import dirname, join, abspath
from tempfile import gettempdir


class JFracturePath:
    MODULE_DIR = dirname(abspath(__file__))
    RSC_DIR = join(MODULE_DIR, 'resources')
    SHARED_DIR = join(MODULE_DIR, 'shared')
    SETTINGS_PATH = join(SHARED_DIR, 'settings.json')
    BLEND_PATH = join(RSC_DIR, 'empty.blend')
    SCRIPT_PATH = join(MODULE_DIR, 'script.py')
    TMP_DIR = gettempdir()
    COPY_BUFFER_PATH = join(TMP_DIR, 'coppybuffer%s.blend')
    PASTE_BUFFER_PATH = join(TMP_DIR, 'pastebuffer%s.blend')
