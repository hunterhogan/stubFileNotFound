def create_pass_builder(tm, opt: int = 2, loop_vectorize: bool = False, slp_vectorize: bool = False):
    """
    Create an LLVM pass builder with the desired optimisation level and options.
    """
