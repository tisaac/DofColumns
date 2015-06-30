=======================================================================
DofColumns: PETSc plugin for column-organized discretizations in PCGAMG
=======================================================================

:Author: `Toby Isaac <tisaac@ices.utexas.edu>`_

.. _Toby Isaac <tisaac@ices.utexas.edu>: tisaac@ices.utexas.edu

Building
--------

This package uses (as of June 2015) `PETSc 3.6`_, which can be obtained from
`bitbucket`_.

.. _development version of PETSc: http://www.mcs.anl.gov/research/projects/petsc/developers/index.html#obtaining

.. _bitbucket: http://bitbucket.org/petsc/petsc

::

    make PETSC_DIR=${PETSC_DIR} PETSC_ARCH=${PETSC_ARCH} all [install]

Usage
-----

This package provides solver tools for discretizations where degrees of
freedom of organized into tightly coupled "columns": all of the degrees of
freedom in a column are assigned to the same MPI process, and are numbered
contiguously.  The columns can have different sizes.

To initialize the plugin, call:

.. code-block:: c

    #include <DofColumns.h>

    PetscErrorCode ierr;

    ierr = DofColumnsInitializePackage();CHKERRQ(ierr);

Let ``A`` be a matrix which has column structure.  The plugin expects that
structure to be described by a `PetscSection`_.  Suppose, for example, that
``A`` is distributed across 2 MPI processes: the first process owns two
columns, the second process one.  Suppose the first and last columns contain
10 degrees of freedom, while the second contains 12.  Here is what the setup
of the section looks like:

.. _PetscSection: http://www.mcs.anl.gov/petsc/petsc-current/docs/manualpages/IS/PetscSection.html

.. code-block:: c
  
    MPI_Comm       comm;
    PetscErrorCode ierr;
    PetscSection   columns;
    PetscMPIInt    rank;
    PetscInt       colStart, colEnd;

    comm = PetscObjectComm((PetscObject)A);
    ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);

    // create the section
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)A),&columns);CHKERRQ(ierr);

    if (!rank) {
      // I have two columns: columns 0 and 1, so my chart is [0,2)
      ierr = PetscSectionSetChart(columns,0,2);CHKERRQ(ierr);
      // Column 0 has 10 dofs
      ierr = PetscSectionSetDof(columns,0,10);CHKERRQ(ierr);
      // Column 1 has 12 dofs
      ierr = PetscSectionSetDof(columns,1,12);CHKERRQ(ierr);

      ierr = PetscSectionSetUp(columns);CHKERRQ(ierr);

      // Column 0 contains dofs [0, 10)
      ierr = PetscSectionSetOffset(columns,0,0);CHKERRQ(ierr);
      // Column 1 contains dofs [10, 22)
      ierr = PetscSectionSetOffset(columns,1,10);CHKERRQ(ierr);
    }
    else {
      // I have one column: column 2, so my chart is [2,3)
      ierr = PetscSectionSetChart(columns,2,3);CHKERRQ(ierr);
      // Column 2 has 10 dofs
      ierr = PetscSectionSetDof(columns,2,12);CHKERRQ(ierr);

      ierr = PetscSectionSetUp(columns);CHKERRQ(ierr);

      // Column 2 contains dofs [22, 32)
      ierr = PetscSectionSetOffset(columns,2,22);CHKERRQ(ierr);
    }

After you have constructed the section, attach it to the matrix:

.. code-block:: c

    ierr = MatSetDofColumns(A,columns);CHKERRQ(ierr);
    // A maintains a reference to columns, so we can safely destroy it
    ierr = PetscSectionDestroy(&columns);CHKERRQ(ierr);

**Note:** If the matrix ``A`` is a block matrix (its block size is greater
than 1), then columns should describe the blocks, i.e., when using
``PetscSectionSetDof()`` to describe how big a column is, one should give the
number of blocks, *not* the total number of degrees of freedom.

Example
-------

Make the test executable with:

::

    make test

The test executable is adapted from a PETSc example: it discretizes and solves
a loading problem for an elastic material in a cubic domain, with user-defined
material parameters and a user defined anisotropy. The problem is discretized
with :math:`n_e` trilinear hexahedral finite elements in each direction. The
height of the domain in the :math:`z`-direction is scaled by a factor of
:math:`\varepsilon`.   All boundary conditions are natural, except for the
bottom boundary, (:math:`z=0`) which has a *Dirichlet* boundary condition on
the normal component of displacement and a *Robin* boundary with coefficient
:math:`\beta` on the tangential components.

If the domain is anisotropic, then smoothed aggregation algebraic multigrid
(SA-AMG) with a pointwise smoother, such as symmetric Gauss-Seidel, may be
inefficient.  To demonstrate this, run

::

    make test_sor

This solves the problem to a fixed tolerance for fixed material coefficients,
:math:`n_e=11` and :math:`\varepsilon=\{1.0,0.1,0.01\}`: in my tests,
the number of Krylov iterations to convergence are 10, 15, and 67, respectively.

Now we repeat the same test, except that we use an incomplete Cholesky
factorization smoother instead of symmetric Gauss-Seidel,

::

    make test_icc

In my tests, the problems with :math:`\varepsilon=\{1.0,0.1,0.01\}` are
now solved in 9, 9, and 6 iterations, respectively.

The column organization is important here: for similar problems that are easier
to analyze (scalar elliptic, 7-point stencil, using geometric multigrid instead
of SA-AMG), one can prove that the effectiveness of an incomplete factorization
smoother should be independent of :math:`\varepsilon`. That is, *if* the
degrees of freedom are ordered properly: with the tightly-coupled degrees of
freedom within a column numbered contiguously.

Unlike geometric multigrid, the coarse grids generated by SA-AMG may not have
the same column organization as the fine grid.  PETSc's default aggregation
strategy (based on a randomized maximal independent set (MIS) algorithm) will
definitely not preserve the column structure.  How does this affect
performance?  To test the same problem with :math:`\varepsilon=0.01` and
:math:`n_e=\{11,23,47\}`, run

::

    make test_h

In my tests, these problems are solved in 6, 11, and 24 iterations,
respectively.  We see that the lack of column structure on the coarse grids
affects the convergence as the mesh size :math:`h=n_e^{-1}` decreases.

Now, to test the same problems, but using coarse grids generated by the
DofColumns plugin, run

::

    make test_h_dofcol

The only difference between this and the previous test is the command line
option ``-pc_gamg_type dofcol``.  Now the problems are solved in 6, 9, and 15
iterations, respectively.  While this is not true :math:`h`-independence, it is
closer.

The advantage of the DofColumns plugin is more pronounced on more difficult
problems.  In the previous examples, the Robin boundary condition coefficient
was chose to be :math:`\beta=1`: the strength of this boundary condition masked
deficiencies in the preconditioners.  For an anisotropic discretization, there
are displacements that are relatively high-frequency in the :math:`x`- and
:math:`y`-directions (thus poorly represented on the coarse grid), but which
are low-energy (in the ``A``-norm) when :math:`\beta` is small, say
:math:`\beta=0.01`.  This case is a more difficult test of the
smoother/hierarchy compatibility.  So, to test the convergence of the multigrid
preconditioners with and without the DofColumns plugin on problems with
:math:`\varepsilon=0.01`, :math:`\beta=0.01` and :math:`n_e=\{8,17,35\}`, run

::

    make test_h_weak

and

::

    make test_h_weak_dofcol

In my tests using the default SA-AMG aggregation, the problems are solved in
16, 38, and 77 iterations, respectively; using the DofColumns aggregation, the
problems are solved in 12, 16, and 20 iterations.


