from _typeshed import Incomplete
from ctypes import Structure
from enum import IntFlag
from llvmlite.binding import ffi as ffi
from llvmlite.binding.common import _encode_string as _encode_string
from llvmlite.binding.initfini import llvm_version_info as llvm_version_info
from typing import NamedTuple

llvm_version_major: Incomplete

class _prunestats(NamedTuple):
    basicblock: Incomplete
    diamond: Incomplete
    fanout: Incomplete
    fanout_raise: Incomplete

class PruneStats(_prunestats):
    """ Holds statistics from reference count pruning.
    """
    def __add__(self, other): ...
    def __sub__(self, other): ...

class _c_PruneStats(Structure):
    _fields_: Incomplete

def dump_refprune_stats(printout: bool = False):
    """ Returns a namedtuple containing the current values for the refop pruning
    statistics. If kwarg `printout` is True the stats are printed to stderr,
    default is False.
    """
def set_time_passes(enable) -> None:
    """Enable or disable the pass timers.

    Parameters
    ----------
    enable : bool
        Set to True to enable the pass timers.
        Set to False to disable the pass timers.
    """
def report_and_reset_timings():
    """Returns the pass timings report and resets the LLVM internal timers.

    Pass timers are enabled by ``set_time_passes()``. If the timers are not
    enabled, this function will return an empty string.

    Returns
    -------
    res : str
        LLVM generated timing report.
    """
def create_module_pass_manager(): ...
def create_function_pass_manager(module): ...

class RefPruneSubpasses(IntFlag):
    PER_BB = 1
    DIAMOND = 2
    FANOUT = 4
    FANOUT_RAISE = 8
    ALL = PER_BB | DIAMOND | FANOUT | FANOUT_RAISE

class PassManager(ffi.ObjectRef):
    """PassManager
    """
    def _dispose(self) -> None: ...
    def add_aa_eval_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#aa-eval-exhaustive-alias-analysis-precision-evaluator

        LLVM 14: `llvm::createAAEvalPass`
        """
    def add_basic_aa_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#basic-aa-basic-alias-analysis-stateless-aa-impl

        LLVM 14: `llvm::createBasicAAWrapperPass`
        """
    def add_constant_merge_pass(self) -> None:
        """
        See http://llvm.org/docs/Passes.html#constmerge-merge-duplicate-global-constants

        LLVM 14: `LLVMAddConstantMergePass`
        """
    def add_dead_arg_elimination_pass(self) -> None:
        """
        See http://llvm.org/docs/Passes.html#deadargelim-dead-argument-elimination

        LLVM 14: `LLVMAddDeadArgEliminationPass`
        """
    def add_dependence_analysis_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#da-dependence-analysis

        LLVM 14: `llvm::createDependenceAnalysisWrapperPass`
        """
    def add_dot_call_graph_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#dot-callgraph-print-call-graph-to-dot-file

        LLVM 14: `llvm::createCallGraphDOTPrinterPass`
        """
    def add_dot_cfg_printer_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#dot-cfg-print-cfg-of-function-to-dot-file

        LLVM 14: `llvm::createCFGPrinterLegacyPassPass`
        """
    def add_dot_dom_printer_pass(self, show_body: bool = False) -> None:
        """
        See https://llvm.org/docs/Passes.html#dot-dom-print-dominance-tree-of-function-to-dot-file

        LLVM 14: `llvm::createDomPrinterPass` and `llvm::createDomOnlyPrinterPass`
        """
    def add_dot_postdom_printer_pass(self, show_body: bool = False) -> None:
        """
        See https://llvm.org/docs/Passes.html#dot-postdom-print-postdominance-tree-of-function-to-dot-file

        LLVM 14: `llvm::createPostDomPrinterPass` and `llvm::createPostDomOnlyPrinterPass`
        """
    def add_globals_mod_ref_aa_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#globalsmodref-aa-simple-mod-ref-analysis-for-globals

        LLVM 14: `llvm::createGlobalsAAWrapperPass`
        """
    def add_iv_users_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#iv-users-induction-variable-users

        LLVM 14: `llvm::createIVUsersPass`
        """
    def add_lint_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#lint-statically-lint-checks-llvm-ir

        LLVM 14: `llvm::createLintLegacyPassPass`
        """
    def add_lazy_value_info_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#lazy-value-info-lazy-value-information-analysis

        LLVM 14: `llvm::createLazyValueInfoPass`
        """
    def add_module_debug_info_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#module-debuginfo-decodes-module-level-debug-info

        LLVM 14: `llvm::createModuleDebugInfoPrinterPass`
        """
    def add_region_info_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#regions-detect-single-entry-single-exit-regions

        LLVM 14: `llvm::createRegionInfoPass`
        """
    def add_scalar_evolution_aa_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#scev-aa-scalarevolution-based-alias-analysis

        LLVM 14: `llvm::createSCEVAAWrapperPass`
        """
    def add_aggressive_dead_code_elimination_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#adce-aggressive-dead-code-elimination

        LLVM 14: `llvm::createAggressiveDCEPass`
        """
    def add_always_inliner_pass(self, insert_lifetime: bool = True) -> None:
        """
        See https://llvm.org/docs/Passes.html#always-inline-inliner-for-always-inline-functions

        LLVM 14: `llvm::createAlwaysInlinerLegacyPass`
        """
    def add_arg_promotion_pass(self, max_elements: int = 3) -> None:
        """
        See https://llvm.org/docs/Passes.html#argpromotion-promote-by-reference-arguments-to-scalars

        LLVM 14: `llvm::createArgumentPromotionPass`
        """
    def add_break_critical_edges_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#break-crit-edges-break-critical-edges-in-cfg

        LLVM 14: `llvm::createBreakCriticalEdgesPass`
        """
    def add_dead_store_elimination_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#dse-dead-store-elimination

        LLVM 14: `llvm::createDeadStoreEliminationPass`
        """
    def add_reverse_post_order_function_attrs_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#function-attrs-deduce-function-attributes

        LLVM 14: `llvm::createReversePostOrderFunctionAttrsPass`
        """
    def add_function_attrs_pass(self) -> None:
        """
        See http://llvm.org/docs/Passes.html#functionattrs-deduce-function-attributes

        LLVM 14: `LLVMAddFunctionAttrsPass`
        """
    def add_function_inlining_pass(self, threshold) -> None:
        """
        See http://llvm.org/docs/Passes.html#inline-function-integration-inlining

        LLVM 14: `createFunctionInliningPass`
        """
    def add_global_dce_pass(self) -> None:
        """
        See http://llvm.org/docs/Passes.html#globaldce-dead-global-elimination

        LLVM 14: `LLVMAddGlobalDCEPass`
        """
    def add_global_optimizer_pass(self) -> None:
        """
        See http://llvm.org/docs/Passes.html#globalopt-global-variable-optimizer

        LLVM 14: `LLVMAddGlobalOptimizerPass`
        """
    def add_ipsccp_pass(self) -> None:
        """
        See http://llvm.org/docs/Passes.html#ipsccp-interprocedural-sparse-conditional-constant-propagation

        LLVM 14: `LLVMAddIPSCCPPass`
        """
    def add_dead_code_elimination_pass(self) -> None:
        """
        See http://llvm.org/docs/Passes.html#dce-dead-code-elimination
        LLVM 14: `llvm::createDeadCodeEliminationPass`
        """
    def add_aggressive_instruction_combining_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#aggressive-instcombine-combine-expression-patterns

        LLVM 14: `llvm::createAggressiveInstCombinerPass`
        """
    def add_internalize_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#internalize-internalize-global-symbols

        LLVM 14: `llvm::createInternalizePass`
        """
    def add_cfg_simplification_pass(self) -> None:
        """
        See http://llvm.org/docs/Passes.html#simplifycfg-simplify-the-cfg

        LLVM 14: `LLVMAddCFGSimplificationPass`
        """
    def add_jump_threading_pass(self, threshold: int = -1) -> None:
        """
        See https://llvm.org/docs/Passes.html#jump-threading-jump-threading

        LLVM 14: `llvm::createJumpThreadingPass`
        """
    def add_lcssa_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#lcssa-loop-closed-ssa-form-pass

        LLVM 14: `llvm::createLCSSAPass`
        """
    def add_gvn_pass(self) -> None:
        """
        See http://llvm.org/docs/Passes.html#gvn-global-value-numbering

        LLVM 14: `LLVMAddGVNPass`
        """
    def add_instruction_combining_pass(self) -> None:
        """
        See http://llvm.org/docs/Passes.html#passes-instcombine

        LLVM 14: `LLVMAddInstructionCombiningPass`
        """
    def add_licm_pass(self) -> None:
        """
        See http://llvm.org/docs/Passes.html#licm-loop-invariant-code-motion

        LLVM 14: `LLVMAddLICMPass`
        """
    def add_loop_deletion_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#loop-deletion-delete-dead-loops

        LLVM 14: `llvm::createLoopDeletionPass`
        """
    def add_loop_extractor_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#loop-extract-extract-loops-into-new-functions

        LLVM 14: `llvm::createLoopExtractorPass`
        """
    def add_single_loop_extractor_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#loop-extract-single-extract-at-most-one-loop-into-a-new-function

        LLVM 14: `llvm::createSingleLoopExtractorPass`
        """
    def add_sccp_pass(self) -> None:
        """
        See http://llvm.org/docs/Passes.html#sccp-sparse-conditional-constant-propagation

        LLVM 14: `LLVMAddSCCPPass`
        """
    def add_loop_strength_reduce_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#loop-reduce-loop-strength-reduction

        LLVM 14: `llvm::createLoopStrengthReducePass`
        """
    def add_loop_simplification_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#loop-simplify-canonicalize-natural-loops

        LLVM 14: `llvm::createLoopSimplifyPass`
        """
    def add_loop_unroll_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#loop-unroll-unroll-loops

        LLVM 14: `LLVMAddLoopUnrollPass`
        """
    def add_loop_unroll_and_jam_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#loop-unroll-and-jam-unroll-and-jam-loops

        LLVM 14: `LLVMAddLoopUnrollAndJamPass`
        """
    def add_loop_unswitch_pass(self, optimize_for_size: bool = False, has_branch_divergence: bool = False) -> None:
        """
        See https://llvm.org/docs/Passes.html#loop-unswitch-unswitch-loops

        LLVM 14: `llvm::createLoopUnswitchPass`
        LLVM 15: `llvm::createSimpleLoopUnswitchLegacyPass`
        """
    def add_lower_atomic_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#loweratomic-lower-atomic-intrinsics-to-non-atomic-form

        LLVM 14: `llvm::createLowerAtomicPass`
        """
    def add_lower_invoke_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#lowerinvoke-lower-invokes-to-calls-for-unwindless-code-generators

        LLVM 14: `llvm::createLowerInvokePass`
        """
    def add_lower_switch_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#lowerswitch-lower-switchinsts-to-branches

        LLVM 14: `llvm::createLowerSwitchPass`
        """
    def add_memcpy_optimization_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#memcpyopt-memcpy-optimization

        LLVM 14: `llvm::createMemCpyOptPass`
        """
    def add_merge_functions_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#mergefunc-merge-functions

        LLVM 14: `llvm::createMergeFunctionsPass`
        """
    def add_merge_returns_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#mergereturn-unify-function-exit-nodes

        LLVM 14: `llvm::createUnifyFunctionExitNodesPass`
        """
    def add_partial_inlining_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#partial-inliner-partial-inliner

        LLVM 14: `llvm::createPartialInliningPass`
        """
    def add_prune_exception_handling_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#prune-eh-remove-unused-exception-handling-info

        LLVM 14: `llvm::createPruneEHPass`
        """
    def add_reassociate_expressions_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#reassociate-reassociate-expressions

        LLVM 14: `llvm::createReassociatePass`
        """
    def add_demote_register_to_memory_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#rel-lookup-table-converter-relative-lookup-table-converter

        LLVM 14: `llvm::createDemoteRegisterToMemoryPass`
        """
    def add_sroa_pass(self) -> None:
        """
        See http://llvm.org/docs/Passes.html#scalarrepl-scalar-replacement-of-aggregates-dt
        Note that this pass corresponds to the ``opt -sroa`` command-line option,
        despite the link above.

        LLVM 14: `llvm::createSROAPass`
        """
    def add_sink_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#sink-code-sinking

        LLVM 14: `llvm::createSinkingPass`
        """
    def add_strip_symbols_pass(self, only_debug: bool = False) -> None:
        """
        See https://llvm.org/docs/Passes.html#strip-strip-all-symbols-from-a-module

        LLVM 14: `llvm::createStripSymbolsPass`
        """
    def add_strip_dead_debug_info_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#strip-dead-debug-info-strip-debug-info-for-unused-symbols

        LLVM 14: `llvm::createStripDeadDebugInfoPass`
        """
    def add_strip_dead_prototypes_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#strip-dead-prototypes-strip-unused-function-prototypes

        LLVM 14: `llvm::createStripDeadPrototypesPass`
        """
    def add_strip_debug_declare_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#strip-debug-declare-strip-all-llvm-dbg-declare-intrinsics

        LLVM 14: `llvm::createStripDebugDeclarePass`
        """
    def add_strip_nondebug_symbols_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#strip-nondebug-strip-all-symbols-except-dbg-symbols-from-a-module

        LLVM 14: `llvm::createStripNonDebugSymbolsPass`
        """
    def add_tail_call_elimination_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#tailcallelim-tail-call-elimination

        LLVM 14: `llvm::createTailCallEliminationPass`
        """
    def add_type_based_alias_analysis_pass(self) -> None:
        """
        LLVM 14: `LLVMAddTypeBasedAliasAnalysisPass`
        """
    def add_basic_alias_analysis_pass(self) -> None:
        """
        See http://llvm.org/docs/AliasAnalysis.html#the-basicaa-pass

        LLVM 14: `LLVMAddBasicAliasAnalysisPass`
        """
    def add_loop_rotate_pass(self) -> None:
        """http://llvm.org/docs/Passes.html#loop-rotate-rotate-loops."""
    def add_target_library_info(self, triple) -> None: ...
    def add_instruction_namer_pass(self) -> None:
        """
        See https://llvm.org/docs/Passes.html#instnamer-assign-names-to-anonymous-instructions.

        LLVM 14: `llvm::createInstructionNamerPass`
        """
    def add_refprune_pass(self, subpasses_flags=..., subgraph_limit: int = 1000) -> None:
        """Add Numba specific Reference count pruning pass.

        Parameters
        ----------
        subpasses_flags : RefPruneSubpasses
            A bitmask to control the subpasses to be enabled.
        subgraph_limit : int
            Limit the fanout pruners to working on a subgraph no bigger than
            this number of basic-blocks to avoid spending too much time in very
            large graphs. Default is 1000. Subject to change in future
            versions.
        """

class ModulePassManager(PassManager):
    def __init__(self, ptr: Incomplete | None = None) -> None: ...
    def run(self, module, remarks_file: Incomplete | None = None, remarks_format: str = 'yaml', remarks_filter: str = ''):
        """
        Run optimization passes on the given module.

        Parameters
        ----------
        module : llvmlite.binding.ModuleRef
            The module to be optimized inplace
        remarks_file : str; optional
            If not `None`, it is the file to store the optimization remarks.
        remarks_format : str; optional
            The format to write; YAML is default
        remarks_filter : str; optional
            The filter that should be applied to the remarks output.
        """
    def run_with_remarks(self, module, remarks_format: str = 'yaml', remarks_filter: str = ''):
        """
        Run optimization passes on the given module and returns the result and
        the remarks data.

        Parameters
        ----------
        module : llvmlite.binding.ModuleRef
            The module to be optimized
        remarks_format : str
            The remarks output; YAML is the default
        remarks_filter : str; optional
            The filter that should be applied to the remarks output.
        """

class FunctionPassManager(PassManager):
    _module: Incomplete
    def __init__(self, module) -> None: ...
    def initialize(self):
        """
        Initialize the FunctionPassManager.  Returns True if it produced
        any changes (?).
        """
    def finalize(self):
        """
        Finalize the FunctionPassManager.  Returns True if it produced
        any changes (?).
        """
    def run(self, function, remarks_file: Incomplete | None = None, remarks_format: str = 'yaml', remarks_filter: str = ''):
        """
        Run optimization passes on the given function.

        Parameters
        ----------
        function : llvmlite.binding.FunctionRef
            The function to be optimized inplace
        remarks_file : str; optional
            If not `None`, it is the file to store the optimization remarks.
        remarks_format : str; optional
            The format of the remarks file; the default is YAML
        remarks_filter : str; optional
            The filter that should be applied to the remarks output.
        """
    def run_with_remarks(self, function, remarks_format: str = 'yaml', remarks_filter: str = ''):
        """
        Run optimization passes on the given function and returns the result
        and the remarks data.

        Parameters
        ----------
        function : llvmlite.binding.FunctionRef
            The function to be optimized inplace
        remarks_format : str; optional
            The format of the remarks file; the default is YAML
        remarks_filter : str; optional
            The filter that should be applied to the remarks output.
        """
