# automated_reasoning_final_project

Omri Isac
Shmuel Orenstein


# Final Project of course Automated Reasoning of Software 67532

The project consists of the following modules:

SMTSolver.py - The main module of the project.
SATSolver.py - A module of the SATSolver
UFSolver.py - A module for the UFSolver
LPSolver.py - A module for the LPSolver
FormulaProcessor.py - A module for processing formulas of BooleanOperator form.

In addition, the project contains two directories of cnf files:
SAT_examples
UNSAT_examples

To run a specific solver, run py SMTSolver.py [theory], where theory is one of {"boolean", "uf", "lp"}
each of these options runs tests over hardcoded formulas or using the cnf files from the directories.

Test functions and hardcoded formulas are written in the SMTSolver.py file.

# Assumptions

	For all parts
	1. Input is assumed to be a BooleanOperator formula, with atomics carrying a string with theory signature (or anything in case of propositional theory).

	SAT Assumptions
	1. Input can be parsed from a cnf file, without first converted to formula (and without applying Tseitin transformation).
	2. Note: BooleanOperator formula is converted using Tseitin transformation.

	UF assumption
	1. All atomics are of equality form, inequality is expressed using the Not class.
	2. Atomic string are syntactically correct and accurate, with all function in their right arity.

	LP assumptions
	1. Atomics have Tq formulas as their strings.
	2. Input adheres the standard form, with correct dimensions of input objects (all variables are assumed to be >= 0) so no strict inequalities appear.
	3. No variable named v in the formula - as it is used for conversion.
	4. Variables names don't contain numbers (e.g. no x0)
	5. All relevant variables are included in all clauses, with the same order.
	6. All coefficients are stated explicitly (including +-1, 0).
	7. Unless inserted as an argument, objective function is the all-ones vector.




