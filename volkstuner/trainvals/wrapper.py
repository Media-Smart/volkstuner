from  .. import engine as ag


def trainval_wrapper(trainval_func, cfg: dict):
    ''' trainval_wrapper
    Args:

    Returns:
        
    '''
    return ag.args(**cfg)(trainval_func)
