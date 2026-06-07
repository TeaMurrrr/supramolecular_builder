import os
import multiprocessing
import warnings
from rdkit import RDLogger

def init_environment(threads=6):
    """Настройка параллелизма и фильтрация предупреждений."""
    RDLogger.DisableLog('rdApp.*')
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Установка лимитов для вычислительных библиотек
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ["MKL_NUM_THREADS"] = str(threads)
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["OMP_STACKSIZE"] = "1G"

def patch_gfn0_path():
    """Автоматический поиск и настройка путей к параметрам GFN0-xTB."""
    possible_paths = []
    if "CONDA_PREFIX" in os.environ:
        possible_paths.append(os.path.join(os.environ["CONDA_PREFIX"], "share", "xtb"))
        possible_paths.append(os.path.join(os.environ["CONDA_PREFIX"], "share"))
    possible_paths.extend(["/usr/local/share/xtb", "/usr/share/xtb", os.getcwd()])

    for path in possible_paths:
        if os.path.exists(os.path.join(path, "param_gfn0-xtb.txt")):
            os.environ["XTBPATH"] = path
            return True
    return False