from .algebras import Quaternion as Quaternion
from .assumptions import (
	AppliedPredicate as AppliedPredicate, ask as ask, assuming as assuming, AssumptionsContext as AssumptionsContext,
	Predicate as Predicate, Q as Q, refine as refine, register_handler as register_handler,
	remove_handler as remove_handler)
from .calculus import (
	AccumBounds as AccumBounds, apply_finite_diff as apply_finite_diff, differentiate_finite as differentiate_finite,
	euler_equations as euler_equations, finite_diff_weights as finite_diff_weights, is_convex as is_convex,
	is_decreasing as is_decreasing, is_increasing as is_increasing, is_monotonic as is_monotonic,
	is_strictly_decreasing as is_strictly_decreasing, is_strictly_increasing as is_strictly_increasing, maximum as maximum,
	minimum as minimum, not_empty_in as not_empty_in, periodicity as periodicity, singularities as singularities,
	stationary_points as stationary_points)
from .concrete import Product as Product, product as product, Sum as Sum, summation as summation
from .core import (
	Add as Add, AlgebraicNumber as AlgebraicNumber, arity as arity, Atom as Atom, AtomicExpr as AtomicExpr, Basic as Basic,
	bottom_up as bottom_up, cacheit as cacheit, Catalan as Catalan, comp as comp, count_ops as count_ops,
	default_sort_key as default_sort_key, Derivative as Derivative, Dict as Dict, diff as diff, Dummy as Dummy, E as E,
	Eq as Eq, Equality as Equality, EulerGamma as EulerGamma, evalf as evalf, evaluate as evaluate, expand as expand,
	expand_complex as expand_complex, expand_func as expand_func, expand_log as expand_log, expand_mul as expand_mul,
	expand_multinomial as expand_multinomial, expand_power_base as expand_power_base, expand_power_exp as expand_power_exp,
	expand_trig as expand_trig, Expr as Expr, factor_nc as factor_nc, factor_terms as factor_terms, Float as Float,
	Function as Function, FunctionClass as FunctionClass, gcd_terms as gcd_terms, Ge as Ge, GoldenRatio as GoldenRatio,
	GreaterThan as GreaterThan, Gt as Gt, I as I, igcd as igcd, ilcm as ilcm, Integer as Integer,
	integer_log as integer_log, integer_nthroot as integer_nthroot, Lambda as Lambda, Le as Le, LessThan as LessThan,
	Lt as Lt, Mod as Mod, mod_inverse as mod_inverse, Mul as Mul, N as N, nan as nan, Ne as Ne, nfloat as nfloat,
	num_digits as num_digits, Number as Number, NumberSymbol as NumberSymbol, oo as oo, ordered as ordered, pi as pi,
	PoleError as PoleError, postorder_traversal as postorder_traversal, Pow as Pow,
	PrecisionExhausted as PrecisionExhausted, preorder_traversal as preorder_traversal, prod as prod, Rational as Rational,
	RealNumber as RealNumber, Rel as Rel, S as S, seterr as seterr, StrictGreaterThan as StrictGreaterThan,
	StrictLessThan as StrictLessThan, Subs as Subs, Symbol as Symbol, symbols as symbols, sympify as sympify,
	SympifyError as SympifyError, trailing as trailing, TribonacciConstant as TribonacciConstant, Tuple as Tuple,
	Unequality as Unequality, UnevaluatedExpr as UnevaluatedExpr, use as use, var as var, vectorize as vectorize,
	Wild as Wild, WildFunction as WildFunction, zoo as zoo)
from .discrete import (
	convolution as convolution, covering_product as covering_product, fft as fft, fwht as fwht, ifft as ifft,
	ifwht as ifwht, intersecting_product as intersecting_product, intt as intt,
	inverse_mobius_transform as inverse_mobius_transform, mobius_transform as mobius_transform, ntt as ntt)
from .functions import (
	Abs as Abs, acos as acos, acosh as acosh, acot as acot, acoth as acoth, acsc as acsc, acsch as acsch,
	adjoint as adjoint, airyai as airyai, airyaiprime as airyaiprime, airybi as airybi, airybiprime as airybiprime,
	andre as andre, appellf1 as appellf1, arg as arg, asec as asec, asech as asech, asin as asin, asinh as asinh,
	assoc_laguerre as assoc_laguerre, assoc_legendre as assoc_legendre, atan as atan, atan2 as atan2, atanh as atanh,
	bell as bell, bernoulli as bernoulli, besseli as besseli, besselj as besselj, besselk as besselk, bessely as bessely,
	beta as beta, betainc as betainc, betainc_regularized as betainc_regularized, binomial as binomial,
	bspline_basis as bspline_basis, bspline_basis_set as bspline_basis_set, carmichael as carmichael, catalan as catalan,
	cbrt as cbrt, ceiling as ceiling, chebyshevt as chebyshevt, chebyshevt_root as chebyshevt_root,
	chebyshevu as chebyshevu, chebyshevu_root as chebyshevu_root, Chi as Chi, Ci as Ci, conjugate as conjugate, cos as cos,
	cosh as cosh, cot as cot, coth as coth, csc as csc, csch as csch, digamma as digamma, DiracDelta as DiracDelta,
	dirichlet_eta as dirichlet_eta, divisor_sigma as divisor_sigma, E1 as E1, Ei as Ei, Eijk as Eijk,
	elliptic_e as elliptic_e, elliptic_f as elliptic_f, elliptic_k as elliptic_k, elliptic_pi as elliptic_pi, erf as erf,
	erf2 as erf2, erf2inv as erf2inv, erfc as erfc, erfcinv as erfcinv, erfi as erfi, erfinv as erfinv, euler as euler,
	exp as exp, exp_polar as exp_polar, expint as expint, factorial as factorial, factorial2 as factorial2,
	FallingFactorial as FallingFactorial, ff as ff, fibonacci as fibonacci, floor as floor, frac as frac,
	fresnelc as fresnelc, fresnels as fresnels, gamma as gamma, gegenbauer as gegenbauer, genocchi as genocchi,
	hankel1 as hankel1, hankel2 as hankel2, harmonic as harmonic, Heaviside as Heaviside, hermite as hermite,
	hermite_prob as hermite_prob, hn1 as hn1, hn2 as hn2, hyper as hyper, Id as Id, im as im,
	interpolating_spline as interpolating_spline, jacobi as jacobi, jacobi_normalized as jacobi_normalized,
	jacobi_symbol as jacobi_symbol, jn as jn, jn_zeros as jn_zeros, kronecker_symbol as kronecker_symbol,
	KroneckerDelta as KroneckerDelta, laguerre as laguerre, LambertW as LambertW, legendre as legendre,
	legendre_symbol as legendre_symbol, lerchphi as lerchphi, LeviCivita as LeviCivita, Li as Li, li as li, ln as ln,
	log as log, loggamma as loggamma, lowergamma as lowergamma, lucas as lucas, marcumq as marcumq, mathieuc as mathieuc,
	mathieucprime as mathieucprime, mathieus as mathieus, mathieusprime as mathieusprime, Max as Max, meijerg as meijerg,
	Min as Min, mobius as mobius, motzkin as motzkin, multigamma as multigamma, partition as partition,
	periodic_argument as periodic_argument, Piecewise as Piecewise, piecewise_exclusive as piecewise_exclusive,
	piecewise_fold as piecewise_fold, polar_lift as polar_lift, polarify as polarify, polygamma as polygamma,
	polylog as polylog, primenu as primenu, primeomega as primeomega, primepi as primepi,
	principal_branch as principal_branch, re as re, real_root as real_root, reduced_totient as reduced_totient, Rem as Rem,
	rf as rf, riemann_xi as riemann_xi, RisingFactorial as RisingFactorial, root as root, sec as sec, sech as sech,
	Shi as Shi, Si as Si, sign as sign, sin as sin, sinc as sinc, SingularityFunction as SingularityFunction, sinh as sinh,
	sqrt as sqrt, stieltjes as stieltjes, subfactorial as subfactorial, tan as tan, tanh as tanh, totient as totient,
	transpose as transpose, tribonacci as tribonacci, trigamma as trigamma, unbranched_argument as unbranched_argument,
	unpolarify as unpolarify, uppergamma as uppergamma, yn as yn, Ynm as Ynm, Ynm_c as Ynm_c, zeta as zeta, Znm as Znm)
from .geometry import (
	are_similar as are_similar, centroid as centroid, Circle as Circle, closest_points as closest_points,
	convex_hull as convex_hull, Curve as Curve, deg as deg, Ellipse as Ellipse, farthest_points as farthest_points,
	GeometryError as GeometryError, idiff as idiff, intersection as intersection, Line as Line, Line2D as Line2D,
	Line3D as Line3D, Parabola as Parabola, Plane as Plane, Point as Point, Point2D as Point2D, Point3D as Point3D,
	Polygon as Polygon, rad as rad, Ray as Ray, Ray2D as Ray2D, Ray3D as Ray3D, RegularPolygon as RegularPolygon,
	Segment as Segment, Segment2D as Segment2D, Segment3D as Segment3D, Triangle as Triangle)
from .integrals import (
	cosine_transform as cosine_transform, CosineTransform as CosineTransform, fourier_transform as fourier_transform,
	FourierTransform as FourierTransform, hankel_transform as hankel_transform, HankelTransform as HankelTransform,
	Integral as Integral, integrate as integrate, inverse_cosine_transform as inverse_cosine_transform,
	inverse_fourier_transform as inverse_fourier_transform, inverse_hankel_transform as inverse_hankel_transform,
	inverse_laplace_transform as inverse_laplace_transform, inverse_mellin_transform as inverse_mellin_transform,
	inverse_sine_transform as inverse_sine_transform, InverseCosineTransform as InverseCosineTransform,
	InverseFourierTransform as InverseFourierTransform, InverseHankelTransform as InverseHankelTransform,
	InverseLaplaceTransform as InverseLaplaceTransform, InverseMellinTransform as InverseMellinTransform,
	InverseSineTransform as InverseSineTransform, laplace_correspondence as laplace_correspondence,
	laplace_initial_conds as laplace_initial_conds, laplace_transform as laplace_transform,
	LaplaceTransform as LaplaceTransform, line_integrate as line_integrate, mellin_transform as mellin_transform,
	MellinTransform as MellinTransform, sine_transform as sine_transform, SineTransform as SineTransform,
	singularityintegrate as singularityintegrate)
from .interactive import (
	init_printing as init_printing, init_session as init_session, interactive_traversal as interactive_traversal)
from .logic import (
	And as And, bool_map as bool_map, Equivalent as Equivalent, false as false, Implies as Implies, ITE as ITE,
	Nand as Nand, Nor as Nor, Not as Not, Or as Or, POSform as POSform, satisfiable as satisfiable,
	simplify_logic as simplify_logic, SOPform as SOPform, to_cnf as to_cnf, to_dnf as to_dnf, to_nnf as to_nnf,
	true as true, Xor as Xor)
from .matrices import (
	Adjoint as Adjoint, banded as banded, block_collapse as block_collapse, blockcut as blockcut,
	BlockDiagMatrix as BlockDiagMatrix, BlockMatrix as BlockMatrix, casoratian as casoratian,
	DeferredVector as DeferredVector, det as det, Determinant as Determinant, diag as diag, DiagMatrix as DiagMatrix,
	diagonalize_vector as diagonalize_vector, DiagonalMatrix as DiagonalMatrix, DiagonalOf as DiagonalOf,
	DotProduct as DotProduct, eye as eye, FunctionMatrix as FunctionMatrix, GramSchmidt as GramSchmidt,
	hadamard_product as hadamard_product, HadamardPower as HadamardPower, HadamardProduct as HadamardProduct,
	hessian as hessian, Identity as Identity, ImmutableDenseMatrix as ImmutableDenseMatrix,
	ImmutableMatrix as ImmutableMatrix, ImmutableSparseMatrix as ImmutableSparseMatrix, Inverse as Inverse,
	jordan_cell as jordan_cell, kronecker_product as kronecker_product, KroneckerProduct as KroneckerProduct,
	list2numpy as list2numpy, MatAdd as MatAdd, MatMul as MatMul, MatPow as MatPow, Matrix as Matrix,
	matrix2numpy as matrix2numpy, matrix_multiply_elementwise as matrix_multiply_elementwise,
	matrix_symbols as matrix_symbols, MatrixBase as MatrixBase, MatrixExpr as MatrixExpr, MatrixPermute as MatrixPermute,
	MatrixSlice as MatrixSlice, MatrixSymbol as MatrixSymbol, MutableDenseMatrix as MutableDenseMatrix,
	MutableMatrix as MutableMatrix, MutableSparseMatrix as MutableSparseMatrix,
	NonSquareMatrixError as NonSquareMatrixError, OneMatrix as OneMatrix, ones as ones, per as per, Permanent as Permanent,
	PermutationMatrix as PermutationMatrix, randMatrix as randMatrix, rot_axis1 as rot_axis1, rot_axis2 as rot_axis2,
	rot_axis3 as rot_axis3, rot_ccw_axis1 as rot_ccw_axis1, rot_ccw_axis2 as rot_ccw_axis2, rot_ccw_axis3 as rot_ccw_axis3,
	rot_givens as rot_givens, ShapeError as ShapeError, SparseMatrix as SparseMatrix, symarray as symarray, Trace as Trace,
	trace as trace, Transpose as Transpose, wronskian as wronskian, ZeroMatrix as ZeroMatrix, zeros as zeros)
from .ntheory import (
	abundance as abundance, binomial_coefficients as binomial_coefficients,
	binomial_coefficients_list as binomial_coefficients_list, composite as composite, compositepi as compositepi,
	continued_fraction as continued_fraction, continued_fraction_convergents as continued_fraction_convergents,
	continued_fraction_iterator as continued_fraction_iterator, continued_fraction_periodic as continued_fraction_periodic,
	continued_fraction_reduce as continued_fraction_reduce, cycle_length as cycle_length, discrete_log as discrete_log,
	divisor_count as divisor_count, divisors as divisors, egyptian_fraction as egyptian_fraction,
	factor_cache as factor_cache, factorint as factorint, factorrat as factorrat, is_abundant as is_abundant,
	is_amicable as is_amicable, is_carmichael as is_carmichael, is_deficient as is_deficient,
	is_mersenne_prime as is_mersenne_prime, is_nthpow_residue as is_nthpow_residue, is_perfect as is_perfect,
	is_primitive_root as is_primitive_root, is_quad_residue as is_quad_residue, isprime as isprime,
	mersenne_prime_exponent as mersenne_prime_exponent, multinomial_coefficients as multinomial_coefficients,
	multiplicity as multiplicity, n_order as n_order, nextprime as nextprime, npartitions as npartitions,
	nthroot_mod as nthroot_mod, perfect_power as perfect_power, pollard_pm1 as pollard_pm1, pollard_rho as pollard_rho,
	prevprime as prevprime, prime as prime, primefactors as primefactors, primerange as primerange,
	primitive_root as primitive_root, primorial as primorial, proper_divisor_count as proper_divisor_count,
	proper_divisors as proper_divisors, quadratic_congruence as quadratic_congruence,
	quadratic_residues as quadratic_residues, randprime as randprime, Sieve as Sieve, sieve as sieve, sqrt_mod as sqrt_mod,
	sqrt_mod_iter as sqrt_mod_iter)
from .parsing import parse_expr as parse_expr
from .plotting import (
	plot as plot, plot_backends as plot_backends, plot_implicit as plot_implicit, plot_parametric as plot_parametric,
	textplot as textplot)
from .polys import (
	AlgebraicField as AlgebraicField, all_roots as all_roots, apart as apart, apart_list as apart_list,
	assemble_partfrac_list as assemble_partfrac_list, BasePolynomialError as BasePolynomialError, cancel as cancel,
	CC as CC, chebyshevt_poly as chebyshevt_poly, chebyshevu_poly as chebyshevu_poly, CoercionFailed as CoercionFailed,
	cofactors as cofactors, ComplexField as ComplexField, ComplexRootOf as ComplexRootOf, compose as compose,
	ComputationFailed as ComputationFailed, construct_domain as construct_domain, content as content,
	count_roots as count_roots, CRootOf as CRootOf, cyclotomic_poly as cyclotomic_poly, decompose as decompose,
	degree as degree, degree_list as degree_list, discriminant as discriminant, div as div, Domain as Domain,
	DomainError as DomainError, EvaluationFailed as EvaluationFailed, EX as EX, ExactQuotientFailed as ExactQuotientFailed,
	ExpressionDomain as ExpressionDomain, exquo as exquo, EXRAW as EXRAW, ExtraneousFactors as ExtraneousFactors,
	factor as factor, factor_list as factor_list, FF as FF, FF_gmpy as FF_gmpy, FF_python as FF_python, field as field,
	field_isomorphism as field_isomorphism, FiniteField as FiniteField, FlagError as FlagError,
	FractionField as FractionField, galois_group as galois_group, gcd as gcd, gcd_list as gcd_list, gcdex as gcdex,
	GeneratorsError as GeneratorsError, GeneratorsNeeded as GeneratorsNeeded, GF as GF, gff as gff, gff_list as gff_list,
	GMPYFiniteField as GMPYFiniteField, GMPYIntegerRing as GMPYIntegerRing, GMPYRationalField as GMPYRationalField,
	grevlex as grevlex, grlex as grlex, groebner as groebner, GroebnerBasis as GroebnerBasis, ground_roots as ground_roots,
	half_gcdex as half_gcdex, hermite_poly as hermite_poly, hermite_prob_poly as hermite_prob_poly,
	HeuristicGCDFailed as HeuristicGCDFailed, HomomorphismFailed as HomomorphismFailed, horner as horner,
	igrevlex as igrevlex, igrlex as igrlex, ilex as ilex, IntegerRing as IntegerRing, interpolate as interpolate,
	interpolating_poly as interpolating_poly, intervals as intervals, invert as invert,
	is_zero_dimensional as is_zero_dimensional, isolate as isolate, IsomorphismFailed as IsomorphismFailed,
	itermonomials as itermonomials, jacobi_poly as jacobi_poly, laguerre_poly as laguerre_poly, LC as LC, lcm as lcm,
	lcm_list as lcm_list, legendre_poly as legendre_poly, lex as lex, LM as LM, LT as LT,
	minimal_polynomial as minimal_polynomial, minpoly as minpoly, monic as monic, Monomial as Monomial,
	MultivariatePolynomialError as MultivariatePolynomialError, NotAlgebraic as NotAlgebraic,
	NotInvertible as NotInvertible, NotReversible as NotReversible, nroots as nroots,
	nth_power_roots_poly as nth_power_roots_poly, OperationNotSupported as OperationNotSupported,
	OptionError as OptionError, Options as Options, parallel_poly_from_expr as parallel_poly_from_expr, pdiv as pdiv,
	pexquo as pexquo, PolificationFailed as PolificationFailed, Poly as Poly, poly as poly,
	poly_from_expr as poly_from_expr, PolynomialDivisionFailed as PolynomialDivisionFailed,
	PolynomialError as PolynomialError, PolynomialRing as PolynomialRing, pquo as pquo, prem as prem,
	prime_decomp as prime_decomp, prime_valuation as prime_valuation, primitive as primitive,
	primitive_element as primitive_element, PurePoly as PurePoly, PythonFiniteField as PythonFiniteField,
	PythonIntegerRing as PythonIntegerRing, PythonRational as PythonRational, QQ as QQ, QQ_gmpy as QQ_gmpy, QQ_I as QQ_I,
	QQ_python as QQ_python, quo as quo, random_poly as random_poly, rational_interpolate as rational_interpolate,
	RationalField as RationalField, real_roots as real_roots, RealField as RealField, reduced as reduced,
	refine_root as refine_root, RefinementFailed as RefinementFailed, rem as rem, resultant as resultant, ring as ring,
	RootOf as RootOf, rootof as rootof, roots as roots, RootSum as RootSum, round_two as round_two, RR as RR,
	sfield as sfield, sqf as sqf, sqf_list as sqf_list, sqf_norm as sqf_norm, sqf_part as sqf_part, sring as sring,
	sturm as sturm, subresultants as subresultants, swinnerton_dyer_poly as swinnerton_dyer_poly,
	symmetric_poly as symmetric_poly, symmetrize as symmetrize, terms_gcd as terms_gcd, to_number_field as to_number_field,
	together as together, total_degree as total_degree, trunc as trunc, UnificationFailed as UnificationFailed,
	UnivariatePolynomialError as UnivariatePolynomialError, vfield as vfield, viete as viete, vring as vring,
	xfield as xfield, xring as xring, ZZ as ZZ, ZZ_gmpy as ZZ_gmpy, ZZ_I as ZZ_I, ZZ_python as ZZ_python)
from .printing import (
	ccode as ccode, cxxcode as cxxcode, dotprint as dotprint, fcode as fcode, glsl_code as glsl_code, jscode as jscode,
	julia_code as julia_code, latex as latex, maple_code as maple_code, mathematica_code as mathematica_code,
	mathml as mathml, multiline_latex as multiline_latex, octave_code as octave_code, pager_print as pager_print,
	pprint as pprint, pprint_try_use_unicode as pprint_try_use_unicode, pprint_use_unicode as pprint_use_unicode,
	pretty as pretty, pretty_print as pretty_print, preview as preview, print_ccode as print_ccode,
	print_fcode as print_fcode, print_glsl as print_glsl, print_gtk as print_gtk, print_jscode as print_jscode,
	print_latex as print_latex, print_maple_code as print_maple_code, print_mathml as print_mathml,
	print_python as print_python, print_rcode as print_rcode, print_tree as print_tree, pycode as pycode, python as python,
	rcode as rcode, rust_code as rust_code, smtlib_code as smtlib_code, srepr as srepr, sstr as sstr, sstrrepr as sstrrepr,
	StrPrinter as StrPrinter, TableForm as TableForm)
from .series import (
	approximants as approximants, difference_delta as difference_delta, EmptySequence as EmptySequence,
	fourier_series as fourier_series, fps as fps, gruntz as gruntz, Limit as Limit, limit as limit, limit_seq as limit_seq,
	O as O, Order as Order, residue as residue, SeqAdd as SeqAdd, SeqFormula as SeqFormula, SeqMul as SeqMul,
	SeqPer as SeqPer, sequence as sequence, series as series)
from .sets import (
	Complement as Complement, Complexes as Complexes, ComplexRegion as ComplexRegion, ConditionSet as ConditionSet,
	Contains as Contains, DisjointUnion as DisjointUnion, EmptySet as EmptySet, FiniteSet as FiniteSet,
	ImageSet as ImageSet, imageset as imageset, Integers as Integers, Intersection as Intersection, Interval as Interval,
	Naturals as Naturals, Naturals0 as Naturals0, OmegaPower as OmegaPower, ord0 as ord0, Ordinal as Ordinal,
	PowerSet as PowerSet, ProductSet as ProductSet, Range as Range, Rationals as Rationals, Reals as Reals, Set as Set,
	SymmetricDifference as SymmetricDifference, Union as Union, UniversalSet as UniversalSet)
from .simplify import (
	besselsimp as besselsimp, collect as collect, collect_const as collect_const, combsimp as combsimp, cse as cse,
	denom as denom, EPath as EPath, epath as epath, exptrigsimp as exptrigsimp, fraction as fraction, FU as FU, fu as fu,
	gammasimp as gammasimp, hyperexpand as hyperexpand, hypersimilar as hypersimilar, hypersimp as hypersimp,
	kroneckersimp as kroneckersimp, logcombine as logcombine, nsimplify as nsimplify, numer as numer, posify as posify,
	powdenest as powdenest, powsimp as powsimp, radsimp as radsimp, ratsimp as ratsimp, ratsimpmodprime as ratsimpmodprime,
	rcollect as rcollect, separatevars as separatevars, signsimp as signsimp, simplify as simplify,
	sqrtdenest as sqrtdenest, trigsimp as trigsimp)
from .solvers import (
	check_assumptions as check_assumptions, checkodesol as checkodesol, checkpdesol as checkpdesol, checksol as checksol,
	classify_ode as classify_ode, classify_pde as classify_pde, decompogen as decompogen, det_quick as det_quick,
	diophantine as diophantine, dsolve as dsolve, factor_system as factor_system,
	failing_assumptions as failing_assumptions, homogeneous_order as homogeneous_order, inv_quick as inv_quick,
	linear_eq_to_matrix as linear_eq_to_matrix, linsolve as linsolve, nonlinsolve as nonlinsolve, nsolve as nsolve,
	ode_order as ode_order, pde_separate as pde_separate, pde_separate_add as pde_separate_add,
	pde_separate_mul as pde_separate_mul, pdsolve as pdsolve, reduce_abs_inequalities as reduce_abs_inequalities,
	reduce_abs_inequality as reduce_abs_inequality, reduce_inequalities as reduce_inequalities, rsolve as rsolve,
	rsolve_hyper as rsolve_hyper, rsolve_poly as rsolve_poly, rsolve_ratio as rsolve_ratio, solve as solve,
	solve_linear as solve_linear, solve_linear_system as solve_linear_system,
	solve_linear_system_LU as solve_linear_system_LU, solve_poly_inequality as solve_poly_inequality,
	solve_poly_system as solve_poly_system, solve_rational_inequalities as solve_rational_inequalities,
	solve_triangulated as solve_triangulated, solve_undetermined_coeffs as solve_undetermined_coeffs,
	solve_univariate_inequality as solve_univariate_inequality, solveset as solveset, substitution as substitution)
from .tensor import (
	Array as Array, DenseNDimArray as DenseNDimArray, derive_by_array as derive_by_array,
	get_contraction_structure as get_contraction_structure, get_indices as get_indices, Idx as Idx,
	ImmutableDenseNDimArray as ImmutableDenseNDimArray, ImmutableSparseNDimArray as ImmutableSparseNDimArray,
	Indexed as Indexed, IndexedBase as IndexedBase, MutableDenseNDimArray as MutableDenseNDimArray,
	MutableSparseNDimArray as MutableSparseNDimArray, NDimArray as NDimArray, permutedims as permutedims, shape as shape,
	SparseNDimArray as SparseNDimArray, tensorcontraction as tensorcontraction, tensordiagonal as tensordiagonal,
	tensorproduct as tensorproduct)
from .utilities import (
	capture as capture, cartes as cartes, dict_merge as dict_merge, filldedent as filldedent, flatten as flatten,
	group as group, has_dups as has_dups, has_variety as has_variety, lambdify as lambdify,
	memoize_property as memoize_property, numbered_symbols as numbered_symbols, postfixes as postfixes,
	prefixes as prefixes, public as public, reshape as reshape, rotations as rotations, sift as sift, subsets as subsets,
	take as take, threaded as threaded, timed as timed, topological_sort as topological_sort, unflatten as unflatten,
	variations as variations, xthreaded as xthreaded)
from _typeshed import Incomplete
from sympy.release import __version__ as __version__

__all__ = ['CC', 'E1', 'EX', 'EXRAW', 'FF', 'FU', 'GF', 'ITE', 'LC', 'LM', 'LT', 'QQ', 'QQ_I', 'RR', 'ZZ', 'ZZ_I', 'Abs', 'AccumBounds', 'Add', 'Adjoint', 'AlgebraicField', 'AlgebraicNumber', 'And', 'AppliedPredicate', 'Array', 'AssumptionsContext', 'Atom', 'AtomicExpr', 'BasePolynomialError', 'Basic', 'BlockDiagMatrix', 'BlockMatrix', 'CRootOf', 'Catalan', 'Chi', 'Ci', 'Circle', 'CoercionFailed', 'Complement', 'ComplexField', 'ComplexRegion', 'ComplexRootOf', 'Complexes', 'ComputationFailed', 'ConditionSet', 'Contains', 'CosineTransform', 'Curve', 'DeferredVector', 'DenseNDimArray', 'Derivative', 'Determinant', 'DiagMatrix', 'DiagonalMatrix', 'DiagonalOf', 'Dict', 'DiracDelta', 'DisjointUnion', 'Domain', 'DomainError', 'DotProduct', 'Dummy', 'E', 'EPath', 'Ei', 'Eijk', 'Ellipse', 'EmptySequence', 'EmptySet', 'Eq', 'Equality', 'Equivalent', 'EulerGamma', 'EvaluationFailed', 'ExactQuotientFailed', 'Expr', 'ExpressionDomain', 'ExtraneousFactors', 'FF_gmpy', 'FF_python', 'FallingFactorial', 'FiniteField', 'FiniteSet', 'FlagError', 'Float', 'FourierTransform', 'FractionField', 'Function', 'FunctionClass', 'FunctionMatrix', 'GMPYFiniteField', 'GMPYIntegerRing', 'GMPYRationalField', 'Ge', 'GeneratorsError', 'GeneratorsNeeded', 'GeometryError', 'GoldenRatio', 'GramSchmidt', 'GreaterThan', 'GroebnerBasis', 'Gt', 'HadamardPower', 'HadamardProduct', 'HankelTransform', 'Heaviside', 'HeuristicGCDFailed', 'HomomorphismFailed', 'I', 'Id', 'Identity', 'Idx', 'ImageSet', 'ImmutableDenseMatrix', 'ImmutableDenseNDimArray', 'ImmutableMatrix', 'ImmutableSparseMatrix', 'ImmutableSparseNDimArray', 'Implies', 'Indexed', 'IndexedBase', 'Integer', 'IntegerRing', 'Integers', 'Integral', 'Intersection', 'Interval', 'Inverse', 'InverseCosineTransform', 'InverseFourierTransform', 'InverseHankelTransform', 'InverseLaplaceTransform', 'InverseMellinTransform', 'InverseSineTransform', 'IsomorphismFailed', 'KroneckerDelta', 'KroneckerProduct', 'Lambda', 'LambertW', 'LaplaceTransform', 'Le', 'LessThan', 'LeviCivita', 'Li', 'Limit', 'Line', 'Line2D', 'Line3D', 'Lt', 'MatAdd', 'MatMul', 'MatPow', 'Matrix', 'MatrixBase', 'MatrixExpr', 'MatrixPermute', 'MatrixSlice', 'MatrixSymbol', 'Max', 'MellinTransform', 'Min', 'Mod', 'Monomial', 'Mul', 'MultivariatePolynomialError', 'MutableDenseMatrix', 'MutableDenseNDimArray', 'MutableMatrix', 'MutableSparseMatrix', 'MutableSparseNDimArray', 'N', 'NDimArray', 'Nand', 'Naturals', 'Naturals0', 'Ne', 'NonSquareMatrixError', 'Nor', 'Not', 'NotAlgebraic', 'NotInvertible', 'NotReversible', 'Number', 'NumberSymbol', 'O', 'OmegaPower', 'OneMatrix', 'OperationNotSupported', 'OptionError', 'Options', 'Or', 'Order', 'Ordinal', 'POSform', 'Parabola', 'Permanent', 'PermutationMatrix', 'Piecewise', 'Plane', 'Point', 'Point2D', 'Point3D', 'PoleError', 'PolificationFailed', 'Poly', 'Polygon', 'PolynomialDivisionFailed', 'PolynomialError', 'PolynomialRing', 'Pow', 'PowerSet', 'PrecisionExhausted', 'Predicate', 'Product', 'ProductSet', 'PurePoly', 'PythonFiniteField', 'PythonIntegerRing', 'PythonRational', 'Q', 'QQ_gmpy', 'QQ_python', 'Quaternion', 'Range', 'Rational', 'RationalField', 'Rationals', 'Ray', 'Ray2D', 'Ray3D', 'RealField', 'RealNumber', 'Reals', 'RefinementFailed', 'RegularPolygon', 'Rel', 'Rem', 'RisingFactorial', 'RootOf', 'RootSum', 'S', 'SOPform', 'Segment', 'Segment2D', 'Segment3D', 'SeqAdd', 'SeqFormula', 'SeqMul', 'SeqPer', 'Set', 'ShapeError', 'Shi', 'Si', 'Sieve', 'SineTransform', 'SingularityFunction', 'SparseMatrix', 'SparseNDimArray', 'StrPrinter', 'StrictGreaterThan', 'StrictLessThan', 'Subs', 'Sum', 'Symbol', 'SymmetricDifference', 'SympifyError', 'TableForm', 'Trace', 'Transpose', 'Triangle', 'TribonacciConstant', 'Tuple', 'Unequality', 'UnevaluatedExpr', 'UnificationFailed', 'Union', 'UnivariatePolynomialError', 'UniversalSet', 'Wild', 'WildFunction', 'Xor', 'Ynm', 'Ynm_c', 'ZZ_gmpy', 'ZZ_python', 'ZeroMatrix', 'Znm', '__version__', 'abundance', 'acos', 'acosh', 'acot', 'acoth', 'acsc', 'acsch', 'adjoint', 'airyai', 'airyaiprime', 'airybi', 'airybiprime', 'algebras', 'all_roots', 'andre', 'apart', 'apart_list', 'appellf1', 'apply_finite_diff', 'approximants', 'are_similar', 'arg', 'arity', 'asec', 'asech', 'asin', 'asinh', 'ask', 'assemble_partfrac_list', 'assoc_laguerre', 'assoc_legendre', 'assuming', 'assumptions', 'atan', 'atan2', 'atanh', 'banded', 'bell', 'bernoulli', 'besseli', 'besselj', 'besselk', 'besselsimp', 'bessely', 'beta', 'betainc', 'betainc_regularized', 'binomial', 'binomial_coefficients', 'binomial_coefficients_list', 'block_collapse', 'blockcut', 'bool_map', 'bottom_up', 'bspline_basis', 'bspline_basis_set', 'cacheit', 'calculus', 'cancel', 'capture', 'carmichael', 'cartes', 'casoratian', 'catalan', 'cbrt', 'ccode', 'ceiling', 'centroid', 'chebyshevt', 'chebyshevt_poly', 'chebyshevt_root', 'chebyshevu', 'chebyshevu_poly', 'chebyshevu_root', 'check_assumptions', 'checkodesol', 'checkpdesol', 'checksol', 'classify_ode', 'classify_pde', 'closest_points', 'cofactors', 'collect', 'collect_const', 'combsimp', 'comp', 'compose', 'composite', 'compositepi', 'concrete', 'conjugate', 'construct_domain', 'content', 'continued_fraction', 'continued_fraction_convergents', 'continued_fraction_iterator', 'continued_fraction_periodic', 'continued_fraction_reduce', 'convex_hull', 'convolution', 'cos', 'cosh', 'cosine_transform', 'cot', 'coth', 'count_ops', 'count_roots', 'covering_product', 'csc', 'csch', 'cse', 'cxxcode', 'cycle_length', 'cyclotomic_poly', 'decompogen', 'decompose', 'default_sort_key', 'deg', 'degree', 'degree_list', 'denom', 'derive_by_array', 'det', 'det_quick', 'diag', 'diagonalize_vector', 'dict_merge', 'diff', 'difference_delta', 'differentiate_finite', 'digamma', 'diophantine', 'dirichlet_eta', 'discrete', 'discrete_log', 'discriminant', 'div', 'divisor_count', 'divisor_sigma', 'divisors', 'doctest', 'dotprint', 'dsolve', 'egyptian_fraction', 'elliptic_e', 'elliptic_f', 'elliptic_k', 'elliptic_pi', 'epath', 'erf', 'erf2', 'erf2inv', 'erfc', 'erfcinv', 'erfi', 'erfinv', 'euler', 'euler_equations', 'evalf', 'evaluate', 'exp', 'exp_polar', 'expand', 'expand_complex', 'expand_func', 'expand_log', 'expand_mul', 'expand_multinomial', 'expand_power_base', 'expand_power_exp', 'expand_trig', 'expint', 'exptrigsimp', 'exquo', 'external', 'eye', 'factor', 'factor_cache', 'factor_list', 'factor_nc', 'factor_system', 'factor_terms', 'factorial', 'factorial2', 'factorint', 'factorrat', 'failing_assumptions', 'false', 'farthest_points', 'fcode', 'ff', 'fft', 'fibonacci', 'field', 'field_isomorphism', 'filldedent', 'finite_diff_weights', 'flatten', 'floor', 'fourier_series', 'fourier_transform', 'fps', 'frac', 'fraction', 'fresnelc', 'fresnels', 'fu', 'functions', 'fwht', 'galois_group', 'gamma', 'gammasimp', 'gcd', 'gcd_list', 'gcd_terms', 'gcdex', 'gegenbauer', 'genocchi', 'geometry', 'get_contraction_structure', 'get_indices', 'gff', 'gff_list', 'glsl_code', 'grevlex', 'grlex', 'groebner', 'ground_roots', 'group', 'gruntz', 'hadamard_product', 'half_gcdex', 'hankel1', 'hankel2', 'hankel_transform', 'harmonic', 'has_dups', 'has_variety', 'hermite', 'hermite_poly', 'hermite_prob', 'hermite_prob_poly', 'hessian', 'hn1', 'hn2', 'homogeneous_order', 'horner', 'hyper', 'hyperexpand', 'hypersimilar', 'hypersimp', 'idiff', 'ifft', 'ifwht', 'igcd', 'igrevlex', 'igrlex', 'ilcm', 'ilex', 'im', 'imageset', 'init_printing', 'init_session', 'integer_log', 'integer_nthroot', 'integrate', 'interactive', 'interactive_traversal', 'interpolate', 'interpolating_poly', 'interpolating_spline', 'intersecting_product', 'intersection', 'intervals', 'intt', 'inv_quick', 'inverse_cosine_transform', 'inverse_fourier_transform', 'inverse_hankel_transform', 'inverse_laplace_transform', 'inverse_mellin_transform', 'inverse_mobius_transform', 'inverse_sine_transform', 'invert', 'is_abundant', 'is_amicable', 'is_carmichael', 'is_convex', 'is_decreasing', 'is_deficient', 'is_increasing', 'is_mersenne_prime', 'is_monotonic', 'is_nthpow_residue', 'is_perfect', 'is_primitive_root', 'is_quad_residue', 'is_strictly_decreasing', 'is_strictly_increasing', 'is_zero_dimensional', 'isolate', 'isprime', 'itermonomials', 'jacobi', 'jacobi_normalized', 'jacobi_poly', 'jacobi_symbol', 'jn', 'jn_zeros', 'jordan_cell', 'jscode', 'julia_code', 'kronecker_product', 'kronecker_symbol', 'kroneckersimp', 'laguerre', 'laguerre_poly', 'lambdify', 'laplace_correspondence', 'laplace_initial_conds', 'laplace_transform', 'latex', 'lcm', 'lcm_list', 'legendre', 'legendre_poly', 'legendre_symbol', 'lerchphi', 'lex', 'li', 'limit', 'limit_seq', 'line_integrate', 'linear_eq_to_matrix', 'linsolve', 'list2numpy', 'ln', 'log', 'logcombine', 'loggamma', 'lowergamma', 'lucas', 'maple_code', 'marcumq', 'mathematica_code', 'mathieuc', 'mathieucprime', 'mathieus', 'mathieusprime', 'mathml', 'matrix2numpy', 'matrix_multiply_elementwise', 'matrix_symbols', 'maximum', 'meijerg', 'mellin_transform', 'memoize_property', 'mersenne_prime_exponent', 'minimal_polynomial', 'minimum', 'minpoly', 'mobius', 'mobius_transform', 'mod_inverse', 'monic', 'motzkin', 'multigamma', 'multiline_latex', 'multinomial_coefficients', 'multipledispatch', 'multiplicity', 'n_order', 'nan', 'nextprime', 'nfloat', 'nonlinsolve', 'not_empty_in', 'npartitions', 'nroots', 'nsimplify', 'nsolve', 'nth_power_roots_poly', 'ntheory', 'nthroot_mod', 'ntt', 'num_digits', 'numbered_symbols', 'numer', 'octave_code', 'ode_order', 'ones', 'oo', 'ord0', 'ordered', 'pager_print', 'parallel_poly_from_expr', 'parse_expr', 'parsing', 'partition', 'pde_separate', 'pde_separate_add', 'pde_separate_mul', 'pdiv', 'pdsolve', 'per', 'perfect_power', 'periodic_argument', 'periodicity', 'permutedims', 'pexquo', 'pi', 'piecewise_exclusive', 'piecewise_fold', 'plot', 'plot_backends', 'plot_implicit', 'plot_parametric', 'plotting', 'polar_lift', 'polarify', 'pollard_pm1', 'pollard_rho', 'poly', 'poly_from_expr', 'polygamma', 'polylog', 'polys', 'posify', 'postfixes', 'postorder_traversal', 'powdenest', 'powsimp', 'pprint', 'pprint_try_use_unicode', 'pprint_use_unicode', 'pquo', 'prefixes', 'prem', 'preorder_traversal', 'pretty', 'pretty_print', 'preview', 'prevprime', 'prime', 'prime_decomp', 'prime_valuation', 'primefactors', 'primenu', 'primeomega', 'primepi', 'primerange', 'primitive', 'primitive_element', 'primitive_root', 'primorial', 'principal_branch', 'print_ccode', 'print_fcode', 'print_glsl', 'print_gtk', 'print_jscode', 'print_latex', 'print_maple_code', 'print_mathml', 'print_python', 'print_rcode', 'print_tree', 'printing', 'prod', 'product', 'proper_divisor_count', 'proper_divisors', 'public', 'pycode', 'python', 'quadratic_congruence', 'quadratic_residues', 'quo', 'rad', 'radsimp', 'randMatrix', 'random_poly', 'randprime', 'rational_interpolate', 'ratsimp', 'ratsimpmodprime', 'rcode', 'rcollect', 're', 'real_root', 'real_roots', 'reduce_abs_inequalities', 'reduce_abs_inequality', 'reduce_inequalities', 'reduced', 'reduced_totient', 'refine', 'refine_root', 'register_handler', 'release', 'rem', 'remove_handler', 'reshape', 'residue', 'resultant', 'rf', 'riemann_xi', 'ring', 'root', 'rootof', 'roots', 'rot_axis1', 'rot_axis2', 'rot_axis3', 'rot_ccw_axis1', 'rot_ccw_axis2', 'rot_ccw_axis3', 'rot_givens', 'rotations', 'round_two', 'rsolve', 'rsolve_hyper', 'rsolve_poly', 'rsolve_ratio', 'rust_code', 'satisfiable', 'sec', 'sech', 'separatevars', 'sequence', 'series', 'seterr', 'sfield', 'shape', 'sieve', 'sift', 'sign', 'signsimp', 'simplify', 'simplify_logic', 'sin', 'sinc', 'sine_transform', 'singularities', 'singularityintegrate', 'sinh', 'smtlib_code', 'solve', 'solve_linear', 'solve_linear_system', 'solve_linear_system_LU', 'solve_poly_inequality', 'solve_poly_system', 'solve_rational_inequalities', 'solve_triangulated', 'solve_undetermined_coeffs', 'solve_univariate_inequality', 'solveset', 'sqf', 'sqf_list', 'sqf_norm', 'sqf_part', 'sqrt', 'sqrt_mod', 'sqrt_mod_iter', 'sqrtdenest', 'srepr', 'sring', 'sstr', 'sstrrepr', 'stationary_points', 'stieltjes', 'strategies', 'sturm', 'subfactorial', 'subresultants', 'subsets', 'substitution', 'summation', 'swinnerton_dyer_poly', 'symarray', 'symbols', 'symmetric_poly', 'symmetrize', 'sympify', 'take', 'tan', 'tanh', 'tensor', 'tensorcontraction', 'tensordiagonal', 'tensorproduct', 'terms_gcd', 'test', 'textplot', 'threaded', 'timed', 'to_cnf', 'to_dnf', 'to_nnf', 'to_number_field', 'together', 'topological_sort', 'total_degree', 'totient', 'trace', 'trailing', 'transpose', 'tribonacci', 'trigamma', 'trigsimp', 'true', 'trunc', 'unbranched_argument', 'unflatten', 'unpolarify', 'uppergamma', 'use', 'utilities', 'var', 'variations', 'vectorize', 'vfield', 'viete', 'vring', 'wronskian', 'xfield', 'xring', 'xthreaded', 'yn', 'zeros', 'zeta', 'zoo']

test: Incomplete
doctest: Incomplete

# Names in __all__ with no definition:
#   algebras
#   assumptions
#   calculus
#   concrete
#   discrete
#   external
#   functions
#   geometry
#   interactive
#   multipledispatch
#   ntheory
#   parsing
#   plotting
#   polys
#   printing
#   release
#   strategies
#   tensor
#   utilities
