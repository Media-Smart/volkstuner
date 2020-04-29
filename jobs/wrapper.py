import  volkstuner.engine as ag


def job_wrapper(job_func, cfg: dict):
    ''' trainval_wrapper
    Args:

    Returns:
        
    '''
    return ag.args(**cfg)(job_func)
