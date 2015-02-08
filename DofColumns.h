#if !defined(DOFCOLUMNS_H)
#define      DOFCOLUMNS_H

#include <petscpc.h>

PETSC_EXTERN PCGAMGType PCGAMGDOFCOL;

PETSC_EXTERN PetscErrorCode DofColumnsInitializePackage(void);
PETSC_EXTERN PetscErrorCode MatSetDofColumns(Mat,PetscSection);
PETSC_EXTERN PetscErrorCode MatGetDofColumns(Mat,PetscSection*);
PETSC_EXTERN PetscErrorCode PCGAMGDofColSetAggregateHeight(PC,PetscInt);
PETSC_EXTERN PetscErrorCode PCGAMGDofColGetAggregateHeight(PC,PetscInt*);
PETSC_EXTERN PetscErrorCode PCGAMGDofColSetColumnGAMGType(PC,PCGAMGType);
PETSC_EXTERN PetscErrorCode PCGAMGDofColGetColumnGAMGType(PC,PCGAMGType*);

#endif    /* DOFCOLUMNS_H */
