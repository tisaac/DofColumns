#include "DofColumns.h"
#include <petscsf.h>
#include <petscmat.h>
#include <petsc-private/matimpl.h>
#include <petscpc.h>
#include <petsc-private/pcgamgimpl.h>

PCGAMGType PCGAMGDOFCOL = "dofcol";

typedef struct _gamgdofcol
{
  PC           columnPC;  /* gamg instance used to manipulate column-reduced systems */
  PetscInt     aggHeight; /* desired minimum height of aggregates within a column */
  PetscBool    useLinear;
}
PC_GAMG_DofCol;

#undef __FUNCT__
#define __FUNCT__ "PCGAMGGetDofCol"
static PetscErrorCode PCGAMGGetDofCol (PC pc, PC_GAMG_DofCol **dofcol)
{
  const char    *type;
  PC_MG         *mg;
  PC_GAMG       *gamg;
  PetscBool      isgamg = PETSC_FALSE;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  type = ((PetscObject)pc)->type_name;
  ierr = PetscStrcmp(type,PCGAMG,&isgamg);CHKERRQ(ierr);
  if (!isgamg) SETERRQ (PetscObjectComm((PetscObject)pc),PETSC_ERR_ARG_WRONG,"pc must be PCGAMG");
  mg      = (PC_MG *) pc->data;
  gamg    = (PC_GAMG *) mg->innerctx;
  *dofcol = (PC_GAMG_DofCol *) gamg->subctx;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGDofColGetSubGAMG"
static PC_GAMG *PCGAMGDofColGetSubGAMG(PC_GAMG_DofCol *dofcol)
{
  PC_MG *mg = (PC_MG *) dofcol->columnPC->data;
  return (PC_GAMG *) mg->innerctx;
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGDofColGiveSubGAMGData"
static PetscErrorCode PCGAMGDofColGiveSubGAMGData(PC pc)
{
  PC_GAMG_DofCol *dofcol;
  PC_GAMG        *gamg, *subgamg;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr    = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  gamg    = (PC_GAMG *) ((PC_MG *) pc->data)->innerctx;
  subgamg = (PC_GAMG *) ((PC_MG *) dofcol->columnPC->data)->innerctx;

  subgamg->data = gamg->data;
  gamg->data    = NULL;

  subgamg->data_cell_cols = gamg->data_cell_cols;
  subgamg->data_cell_rows = gamg->data_cell_rows;
  subgamg->data_sz        = gamg->data_sz;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGDofColTakeSubGAMGData"
static PetscErrorCode PCGAMGDofColTakeSubGAMGData(PC pc)
{
  PC_GAMG_DofCol *dofcol;
  PC_GAMG        *gamg, *subgamg;
  PetscErrorCode  ierr;

  PetscFunctionBeginUser;
  ierr    = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  gamg    = (PC_GAMG *) ((PC_MG *) pc->data)->innerctx;
  subgamg = (PC_GAMG *) ((PC_MG *) dofcol->columnPC->data)->innerctx;

  gamg->data    = subgamg->data;
  subgamg->data = NULL;

  gamg->data_cell_cols = subgamg->data_cell_cols;
  gamg->data_cell_rows = subgamg->data_cell_rows;
  gamg->data_sz        = subgamg->data_sz;
  PetscFunctionReturn(0);
}

/* given a PetscSection, compute a prolongator matrix from points to dofs
 * with a value of 1. for every (dof,point) entry.
 */
#undef __FUNCT__
#define __FUNCT__ "PetscSectionGetDofProlongator"
static PetscErrorCode PetscSectionGetDofProlongator(PetscSection sec,Mat *P)
{
  PetscInt       pStart, pEnd, aStart, aEnd, rStart, rEnd, cStart, cEnd, p, maxDof, *idx;
  PetscScalar    *ones;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = PetscSectionGetChart(sec,&pStart,&pEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetOffsetRange(sec,&aStart,&aEnd);CHKERRQ(ierr);
  ierr = PetscSectionGetMaxDof(sec,&maxDof);CHKERRQ(ierr);
  ierr = MatCreateAIJ(PetscObjectComm((PetscObject)sec),aEnd-aStart,pEnd-pStart,PETSC_DETERMINE,PETSC_DETERMINE,1,NULL,0,NULL,P);CHKERRQ(ierr);
  ierr = MatGetOwnershipRange(*P,&rStart,&rEnd);CHKERRQ(ierr);
  if ((aEnd - aStart) && (rStart != aStart || rEnd != aEnd)) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"The atlas of the PetscSection does not define a valid layout: [%d, %d) != [%d, %d).", rStart, rEnd, aStart, aEnd);
  ierr = MatGetOwnershipRangeColumn(*P,&cStart,&cEnd);CHKERRQ(ierr);
  if (cStart != pStart || cEnd != pEnd) SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_WRONG,"The chart of the PetscSection does not define a valid layout: [%d, %d) != [%d, %d).", cStart, cEnd, pStart, pEnd);
  ierr = PetscMalloc2(maxDof,&ones,maxDof,&idx);CHKERRQ(ierr);
  for (p = 0; p < maxDof; p++) ones[p] = 1.;
  for (p = pStart; p < pEnd; p++) {
    PetscInt dof, off, d;

    ierr = PetscSectionGetDof(sec,p,&dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(sec,p,&off);CHKERRQ(ierr);
    for (d = 0; d < dof; d++) {
      idx[d] = off + d;
    }
    ierr = MatSetValues(*P,dof,idx,1,&p,ones,INSERT_VALUES);CHKERRQ(ierr);
  }
  ierr = MatAssemblyBegin(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = MatAssemblyEnd(*P,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
  ierr = PetscFree2(ones,idx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* compute the adjacency graph that will be used to define aggregates:
 * in this case, we return the adjacency graph of columns */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGGraph_DofCol"
PetscErrorCode PCGAMGGraph_DofCol(PC pc,const Mat Amat,Mat *a_Gmat)
{
  PC_GAMG_DofCol *dofcol;
  PC_GAMG        *subgamg;
  PetscObject    columns_obj;
  PetscSection   columns;
  Mat            fullGmat;
  Mat            colProl;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);
  ierr    = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  subgamg = PCGAMGDofColGetSubGAMG(dofcol);
  ierr    = PetscObjectQuery((PetscObject)Amat,"DofColumns",&columns_obj);CHKERRQ(ierr);
  if (!columns_obj) SETERRQ(PetscObjectComm((PetscObject)Amat),PETSC_ERR_ARG_WRONG,"Dof columns not set: call MatSetDofColumns() on the system matrix.");
  columns = (PetscSection) columns_obj;
  ierr    = PCGAMGDofColGiveSubGAMGData(pc);
  ierr    = subgamg->ops->graph(dofcol->columnPC,Amat,&fullGmat);CHKERRQ(ierr);
  ierr    = PCGAMGDofColTakeSubGAMGData(pc);
  ierr    = PetscSectionGetDofProlongator(columns,&colProl);CHKERRQ(ierr);
  ierr    = MatPtAP(fullGmat,colProl,MAT_INITIAL_MATRIX,1.,a_Gmat);CHKERRQ(ierr);
  ierr    = PetscObjectCompose((PetscObject)(*a_Gmat),"DofColumns",(PetscObject)columns);CHKERRQ(ierr);
  ierr    = PetscObjectCompose((PetscObject)(*a_Gmat),"DofColumns_fullGmat",(PetscObject)fullGmat);CHKERRQ(ierr);
  ierr    = MatDestroy(&fullGmat);CHKERRQ(ierr);
  ierr    = MatDestroy(&colProl);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

/* converts aggCD to local numbering, gets a section/sf relating those local columns to global dofs */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGDofColGetLocalSection"
static PetscErrorCode PCGAMGDofColGetLocalSection(PC pc,PetscSection columns,Mat Gmat,Mat dofGmat,PetscCoarsenData *aggCD,PetscSection *outSec, IS *dofPerm)
{
  MPI_Comm         comm;
  PetscMPIInt      size, rank;
  PetscSection     newSec;
  PetscInt         asiz, a, nloc, nagg, ndir, nlocDof, offset, offsetDof, *globalCols, count, pStartOld, pEndOld, pStartNew, pEndNew, p, thisAgg, *dofVec;
  PetscInt         (*colDofold)[2], (*colDofnew)[2], (*gColToAggOld)[2], (*gColToAggNew)[2];
  PetscLayout      layout;
  PetscSF          colSF, aggSF;
  PetscSFNode     *globalColNodes;
  MPI_Datatype     doubleInt;
  PetscErrorCode   ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(columns,PETSC_SECTION_CLASSID,2);
  comm = PetscObjectComm((PetscObject)columns);
  ierr = MPI_Comm_size(comm,&size);CHKERRQ(ierr);
  ierr = MPI_Comm_rank(comm,&rank);CHKERRQ(ierr);
  ierr = PetscSectionGetChart(columns,&pStartOld,&pEndOld);CHKERRQ(ierr);
  ierr = PetscSectionCreate(comm,&newSec);CHKERRQ(ierr);
  asiz = aggCD->size;
  nloc = 0;
  nagg = 0;
  /* count the number of aggregates and the number of columns aggregated onto
   * this process */
  for (a = 0; a < asiz; a++) {
    PetscInt thisSize;

    ierr  = PetscCDSizeAt(aggCD,a,&thisSize);CHKERRQ(ierr);
    nloc += thisSize;
    nagg += !!thisSize;
  }
  ierr = PetscMalloc1(nloc,&globalCols);CHKERRQ(ierr);
  ierr = PetscMalloc2((pEndOld-pStartOld),&gColToAggOld,nloc,&gColToAggNew);CHKERRQ(ierr);
  for (p = pStartOld; p < pEndOld; p++) gColToAggOld[p - pStartOld][1] = -1;
  /* mark global columns in aggregates */
  for (a = 0, count = 0, thisAgg = 0; a < asiz; a++) {
    PetscBool  isEmpty;
    PetscCDPos pos;

    ierr  = PetscCDEmptyAt(aggCD,a,&isEmpty);CHKERRQ(ierr);
    if (isEmpty) continue;
    ierr = PetscCDGetHeadPos(aggCD,a,&pos);CHKERRQ(ierr);
    while (pos) {
      ierr = PetscLLNGetID(pos, &(globalCols[count]));CHKERRQ(ierr);
      gColToAggNew[count][0] = rank;
      gColToAggNew[count][1] = thisAgg;
      ierr = PetscLLNSetID(pos, count++);CHKERRQ(ierr); /* convert to local ordering */
      ierr = PetscCDGetNextPos(aggCD,a,&pos);CHKERRQ(ierr);
    }
    thisAgg++;
  }
  ierr = MatGetLayouts(Gmat,&layout,NULL);CHKERRQ(ierr);
  ierr = PetscSFCreate(comm,&colSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraphLayout(colSF,layout,nloc,NULL,PETSC_OWN_POINTER,globalCols);CHKERRQ(ierr);
  ierr = PetscFree(globalCols);CHKERRQ(ierr);
  ierr = MPI_Type_contiguous(2,MPIU_INT,&doubleInt);CHKERRQ(ierr);
  ierr = MPI_Type_commit(&doubleInt);CHKERRQ(ierr);
  ierr = PetscSFReduceBegin(colSF,doubleInt,gColToAggNew,gColToAggOld,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFReduceEnd(colSF,doubleInt,gColToAggNew,gColToAggOld,MPI_REPLACE);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&colSF);CHKERRQ(ierr);
  /* count dirichlet columns: for the purposes of communication, we treat them as their own aggregates */
  ierr = PetscMalloc1((pEndOld-pStartOld),&globalColNodes);CHKERRQ(ierr);
  for (p = pStartOld, ndir = 0; p < pEndOld; p++) {
    if (gColToAggOld[p-pStartOld][1] == -1) {
      globalColNodes[p - pStartOld].rank  = rank;
      globalColNodes[p - pStartOld].index = nagg++;
      nloc++;
      ndir++;
    }
    else {
      globalColNodes[p - pStartOld].rank  = gColToAggOld[p - pStartOld][0];
      globalColNodes[p - pStartOld].index = gColToAggOld[p - pStartOld][1];
    }
  }
  ierr      = PetscFree2(gColToAggOld,gColToAggNew);CHKERRQ(ierr);
  ierr      = MPI_Scan(&nloc,&offset,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  offset   -= nloc;
  pStartNew = offset;
  pEndNew   = offset+nloc;
  ierr      = PetscSectionSetChart(newSec,pStartNew,pEndNew);CHKERRQ(ierr);

  ierr = PetscSFCreate(comm,&aggSF);CHKERRQ(ierr);
  ierr = PetscSFSetGraph(aggSF,nagg,(pEndOld-pStartOld),NULL,PETSC_OWN_POINTER,globalColNodes,PETSC_OWN_POINTER);CHKERRQ(ierr);
  ierr = PetscSFSetRankOrder(aggSF,PETSC_TRUE);CHKERRQ(ierr);
  ierr = PetscMalloc2((pEndOld-pStartOld),&colDofold,nloc,&colDofnew);CHKERRQ(ierr);
  for (p = pStartOld; p < pEndOld; p++) {
    PetscInt dof, off;

    ierr = PetscSectionGetDof(columns,p,&dof);CHKERRQ(ierr);
    ierr = PetscSectionGetOffset(columns,p,&off);CHKERRQ(ierr);
    colDofold[p - pStartOld][0] = dof;
    colDofold[p - pStartOld][1] = off;
  }
  ierr = PetscSFGatherBegin(aggSF,doubleInt,colDofold,colDofnew);CHKERRQ(ierr);
  ierr = PetscSFGatherEnd(aggSF,doubleInt,colDofold,colDofnew);CHKERRQ(ierr);
  ierr = MPI_Type_free(&doubleInt);CHKERRQ(ierr);
  ierr = PetscSFDestroy(&aggSF);CHKERRQ(ierr);
  for (p = pStartNew; p < pEndNew; p++) {
    PetscInt dof;

    dof  = colDofnew[p - pStartNew][0];
    ierr = PetscSectionSetDof(newSec,p,dof);CHKERRQ(ierr);
  }
  ierr = PetscSectionSetUp(newSec);CHKERRQ(ierr);
  ierr = PetscSectionGetStorageSize(newSec,&nlocDof);CHKERRQ(ierr);
  ierr = MPI_Scan(&nlocDof,&offsetDof,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  offsetDof -= nlocDof;
  ierr = PetscMalloc1(nlocDof,&dofVec);CHKERRQ(ierr);
  for (p = pStartNew; p < pEndNew; p++) {
    PetscInt off, dof, oldoff, q;

    dof    = colDofnew[p - pStartNew][0];
    oldoff = colDofnew[p - pStartNew][1];
    ierr   = PetscSectionGetOffset(newSec,p,&off);CHKERRQ(ierr);
    ierr   = PetscSectionSetOffset(newSec,p,off+offsetDof);CHKERRQ(ierr);
    for (q = 0; q < dof; q++) {
      dofVec[off + q] = oldoff + q;
    }
  }
  ierr = ISCreateGeneral(comm,nlocDof,dofVec,PETSC_OWN_POINTER,dofPerm);CHKERRQ(ierr);
  ierr = ISSetPermutation(*dofPerm);CHKERRQ(ierr);
  ierr = PetscFree2(colDofold,colDofnew);CHKERRQ(ierr);
  *outSec = newSec;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGAggSectionGetFullAggs_Linear"
static PetscErrorCode PCGAMGAggSectionGetFullAggs_Linear(PC pc,PetscCoarsenData *col_agg,PetscSection colSec,PetscCoarsenData *agg_lists,Mat *Gmat,PetscSection* coarseSec,IS dofPerm)
{
  MPI_Comm       comm;
  PC_GAMG_DofCol *dofcol;
  PetscInt       aggHeight, pStart, pEnd, p, vStart, vEnd, asiz, a, maxAgg;
  PetscInt       *cols, *colHeights;
  PetscInt       naggcols, colaggoff, aggcount;
  PetscSection   crsSec;
  PetscLayout    GmatLayout;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  comm      = PetscObjectComm((PetscObject)pc);
  ierr      = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  aggHeight = dofcol->aggHeight;
  ierr      = PetscSectionGetChart(colSec,&pStart,&pEnd);CHKERRQ(ierr);
  ierr      = PetscSectionGetOffsetRange(colSec,&vStart,&vEnd);CHKERRQ(ierr);
  asiz      = col_agg->size;
  ierr      = PetscSectionCreate(comm,&crsSec);CHKERRQ(ierr);

  maxAgg = 0;
  naggcols = 0;
  for (a = 0; a < asiz; a++) {
    PetscInt size;

    ierr = PetscCDSizeAt(col_agg,a,&size);CHKERRQ(ierr);
    naggcols += !!size;
    maxAgg = PetscMax(size,maxAgg);CHKERRQ(ierr);
  }
  ierr = MPI_Scan(&naggcols,&colaggoff,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
  colaggoff -= naggcols;
  ierr = PetscSectionSetChart(crsSec,colaggoff,colaggoff+naggcols);CHKERRQ(ierr);
  ierr = PetscMalloc2(maxAgg,&cols,maxAgg,&colHeights);CHKERRQ(ierr);
  for (a = 0, aggcount = 0; a < asiz; a++) {
    PetscInt size, thisCount, minHeight, nAggs, k;
    PetscCDPos pos;

    ierr = PetscCDSizeAt(col_agg,a,&size);CHKERRQ(ierr);
    if (!size) continue;
    ierr = PetscCDGetHeadPos(col_agg,a,&pos);CHKERRQ(ierr);
    thisCount = 0;
    minHeight = PETSC_MAX_INT;
    while (pos) {
      PetscInt lid, dof;

      ierr = PetscLLNGetID(pos, &lid);CHKERRQ(ierr);
      ierr = PetscCDGetNextPos(col_agg,a,&pos);CHKERRQ(ierr);
      cols[thisCount] = lid + pStart;
      ierr = PetscSectionGetDof(colSec,lid + pStart,&dof);CHKERRQ(ierr);
      colHeights[thisCount++] = dof;
      minHeight = PetscMin(dof,minHeight);
    }
    nAggs = PetscMax(minHeight / aggHeight,1);
    ierr  = PetscSectionSetDof(crsSec,(aggcount++) + colaggoff,nAggs);CHKERRQ(ierr);
    for (k = 0; k < nAggs; k++) {
      PetscInt thisHead = -1;
      for (p = 0; p < size; p++) {
        PetscInt off;
        PetscInt thisStart = (colHeights[p] * k) / nAggs;
        PetscInt thisEnd   = (colHeights[p] * (k + 1)) / nAggs;
        PetscInt l;

        ierr = PetscSectionGetOffset(colSec,cols[p],&off);CHKERRQ(ierr);
        thisStart += off;
        thisEnd += off;
        if (thisHead == -1) {
          thisHead = thisStart;
        }
        for (l = thisStart; l < thisEnd; l++) {
          ierr = PetscCDAppendID(agg_lists,thisHead - vStart,l);CHKERRQ(ierr);
        }
      }
    }
  }
  ierr = PetscSectionSetUp(crsSec);CHKERRQ(ierr);
  {
    PetscInt cvStart, cvEnd, cvDof, cvOff;
    ierr = PetscSectionGetOffsetRange(crsSec,&cvStart,&cvEnd);CHKERRQ(ierr);
    cvDof = cvEnd - cvStart;
    ierr = MPI_Scan(&cvDof,&cvOff,1,MPIU_INT,MPI_SUM,comm);CHKERRQ(ierr);
    cvOff -= cvDof;
    for (p = colaggoff; p < colaggoff + naggcols; p++) {
      PetscInt off;

      ierr = PetscSectionGetOffset(crsSec,p,&off);CHKERRQ(ierr);
      ierr = PetscSectionSetOffset(crsSec,p,off+cvOff);CHKERRQ(ierr);
    }
  }
  *coarseSec = crsSec;
  ierr = MatGetLayouts(*Gmat,&GmatLayout,NULL);CHKERRQ(ierr);
  {
    PetscSF dofSF;
    const PetscInt *dofVal;
    PetscInt *invVal, *invId;
    Mat oldGmat, newGmatPerm, newGmat;
    PetscInt n, oldn, *dnnz, *onnz, *rowids, *colids;
    PetscScalar *ones;
    IS invPerm;

    n    = vEnd - vStart;
    oldn = GmatLayout->n;
    ierr = ISGetIndices(dofPerm,&dofVal);CHKERRQ(ierr);
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)*Gmat),&dofSF);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(dofSF,GmatLayout,n,NULL,PETSC_OWN_POINTER,dofVal);CHKERRQ(ierr);
    ierr = PetscMalloc1(n,&invId);CHKERRQ(ierr);
    ierr = PetscMalloc1(oldn,&invVal);CHKERRQ(ierr);
    for (a = 0; a < n; a++) {
      invId[a] = a + vStart;
    }
    ierr = PetscSFReduceBegin(dofSF,MPIU_INT,invId,invVal,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFReduceEnd(dofSF,MPIU_INT,invId,invVal,MPI_REPLACE);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&dofSF);CHKERRQ(ierr);
    ierr = PetscFree(invId);CHKERRQ(ierr);
    ierr = ISCreateGeneral(PetscObjectComm((PetscObject)dofPerm),oldn,invVal,PETSC_OWN_POINTER,&invPerm);CHKERRQ(ierr);

    ierr = MatCreate(PetscObjectComm((PetscObject)*Gmat),&newGmatPerm);CHKERRQ(ierr);
    ierr = MatSetType(newGmatPerm,MATAIJ);CHKERRQ(ierr);
    ierr = MatSetSizes(newGmatPerm,n,oldn,GmatLayout->N,GmatLayout->N);CHKERRQ(ierr);
    ierr = PetscMalloc2(n,&dnnz,n,&onnz);CHKERRQ(ierr);
    maxAgg = 0;
    for (a = 0; a < n; a++) {
      if (dofVal[a] < GmatLayout->rstart || dofVal[a] >= GmatLayout->rend) {
        onnz[a] = 1;
        dnnz[a] = 0;
      }
      else {
        onnz[a] = 0;
        dnnz[a] = 1;
      }
    }
    for (a = 0; a < agg_lists->size; a++) {
      PetscInt size;
      PetscCDPos pos;
      PetscInt ddnnz = 0, oonnz = 0;

      ierr = PetscCDSizeAt(agg_lists,a,&size);CHKERRQ(ierr);
      if (!size) continue;

      ierr = PetscCDGetHeadPos(agg_lists,a,&pos);CHKERRQ(ierr);
      maxAgg = PetscMax(maxAgg,size);
      while (pos) {
        PetscInt lid;

        ierr = PetscLLNGetID(pos, &lid);CHKERRQ(ierr);
        ierr = PetscCDGetNextPos(agg_lists,a,&pos);CHKERRQ(ierr);

        if (dofVal[lid - vStart] < GmatLayout->rstart || dofVal[lid - vStart] >= GmatLayout->rend) {
          oonnz++;
        }
        else {
          ddnnz++;
        }
      }
      ierr = PetscCDGetHeadPos(agg_lists,a,&pos);CHKERRQ(ierr);
      while (pos) {
        PetscInt lid;

        ierr = PetscLLNGetID(pos, &lid);CHKERRQ(ierr);
        ierr = PetscCDGetNextPos(agg_lists,a,&pos);CHKERRQ(ierr);

        dnnz[lid - vStart] = ddnnz;
        onnz[lid - vStart] = oonnz;
      }
    }
    oldGmat = *Gmat;
    ierr = MatDestroy(&oldGmat);CHKERRQ(ierr);
    ierr = MatXAIJSetPreallocation(newGmatPerm,1,dnnz,onnz,NULL,NULL);CHKERRQ(ierr);
    ierr = PetscFree2(dnnz,onnz);CHKERRQ(ierr);
    ierr = PetscMalloc3(maxAgg,&rowids,maxAgg,&colids,maxAgg*maxAgg,&ones);CHKERRQ(ierr);
    for (a = 0; a < maxAgg * maxAgg; a++) ones[a] = 1.;
    for (a = 0; a < agg_lists->size; a++) {
      PetscInt size, thisCount;
      PetscCDPos pos;

      ierr = PetscCDSizeAt(agg_lists,a,&size);CHKERRQ(ierr);
      if (!size) continue;

      ierr = PetscCDGetHeadPos(agg_lists,a,&pos);CHKERRQ(ierr);
      thisCount = 0;
      while (pos) {
        PetscInt lid;

        ierr = PetscLLNGetID(pos, &lid);CHKERRQ(ierr);
        ierr = PetscCDGetNextPos(agg_lists,a,&pos);CHKERRQ(ierr);

        rowids[thisCount]   = lid;
        colids[thisCount++] = dofVal[lid-vStart];
      }
      ierr = MatSetValues(newGmatPerm,size,rowids,size,colids,ones,INSERT_VALUES);CHKERRQ(ierr);
    }
    ierr = ISRestoreIndices(dofPerm,&dofVal);CHKERRQ(ierr);
    ierr = PetscFree3(rowids,colids,ones);CHKERRQ(ierr);
    ierr = MatAssemblyBegin(newGmatPerm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(newGmatPerm,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatGetSubMatrix(newGmatPerm,invPerm,NULL,MAT_INITIAL_MATRIX,&newGmat);CHKERRQ(ierr);
    ierr = MatDestroy(&newGmatPerm);CHKERRQ(ierr);
    ierr = ISDestroy(&invPerm);CHKERRQ(ierr);
    *Gmat = newGmat;
  }

  ierr = PetscFree2(cols,colHeights);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGAggSectionGetFullAggs_Full"
static PetscErrorCode PCGAMGAggSectionGetFullAggs_Full(PC pc,PetscCoarsenData *col_agg,PetscSection colSec,PetscCoarsenData *agg_lists,Mat *Gmat,PetscSection* coarseSec,IS dofPerm)
{
  PetscFunctionBeginUser;
  SETERRQ(PetscObjectComm((PetscObject)pc),PETSC_ERR_SUP,"Full column aggregate creation not implemented yet");
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGAggSectionGetFullAggs"
static PetscErrorCode PCGAMGAggSectionGetFullAggs(PC pc,PetscCoarsenData *col_agg,PetscSection colSec,PetscCoarsenData *agg_lists,Mat *Gmat,PetscSection* coarseSec,IS dofPerm)
{
  PC_GAMG_DofCol *dofcol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  if (dofcol->useLinear) {
    ierr = PCGAMGAggSectionGetFullAggs_Linear(pc,col_agg,colSec,agg_lists,Gmat,coarseSec,dofPerm);CHKERRQ(ierr);
  }
  else {
    ierr = PCGAMGAggSectionGetFullAggs_Full(pc,col_agg,colSec,agg_lists,Gmat,coarseSec,dofPerm);CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

/* Call the coarsen routine from the columnPC, then convert Gmat and agg_lists from columns to full */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGCoarsen_DofCol"
PetscErrorCode PCGAMGCoarsen_DofCol(PC pc,Mat *Gmat,PetscCoarsenData **agg_lists)
{
  PC_GAMG_DofCol *dofcol;
  PC_GAMG        *subgamg;
  PetscCoarsenData *sub_agg_lists, *agg3, *agg3_local;
  PetscObject    columns_obj, dofGmat_obj;
  PetscSection   columns, columnsLocal, coarseColumns;
  IS             dofPerm;
  Mat            dofGmat;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr    = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  subgamg = PCGAMGDofColGetSubGAMG(dofcol);
  ierr    = PetscObjectQuery((PetscObject)(*Gmat),"DofColumns",&columns_obj);CHKERRQ(ierr);
  if (!columns_obj) SETERRQ(PetscObjectComm((PetscObject)Gmat),PETSC_ERR_ARG_WRONGSTATE,"Could not get the dof columns from the adjacency matrix");
  columns = (PetscSection) columns_obj;
  ierr    = PetscObjectQuery((PetscObject)(*Gmat),"DofColumns_fullGmat",&dofGmat_obj);CHKERRQ(ierr);
  dofGmat = (Mat) dofGmat_obj;
  ierr    = PetscObjectReference((PetscObject)dofGmat);CHKERRQ(ierr);
  {
    PetscLayout glayout;

    ierr = MatGetLayouts(dofGmat,&glayout,NULL);CHKERRQ(ierr);
    ierr = PetscCDCreate(glayout->n,&agg3);CHKERRQ(ierr);
  }
  ierr = PetscObjectCompose((PetscObject)(*Gmat),"DofColumns_fullGmat",NULL);CHKERRQ(ierr);
  subgamg->firstCoarsen = ((PC_GAMG *) ((PC_MG *) pc->data)->innerctx)->firstCoarsen;
  ierr = PCGAMGDofColGiveSubGAMGData(pc);
  ierr = subgamg->ops->coarsen(dofcol->columnPC,Gmat,&sub_agg_lists);CHKERRQ(ierr); /* get the flattened aggs and modified flattened adjacency matrix*/
  ierr = PCGAMGDofColTakeSubGAMGData(pc);
  /* convert the flattened aggs to local numbering; get a version of the
   * dofGmat that only includes adjacencies within aggregate columns in
   * localized order; get the section describing the local ordering of
   * columns, and a permutation from the localized dofs to the original global
   * dofs */
  ierr = PCGAMGDofColGetLocalSection(pc,columns,*Gmat,dofGmat,sub_agg_lists,&columnsLocal,&dofPerm);CHKERRQ(ierr);
  {
    PetscInt    nLocal;
    Mat colGmat = *Gmat;

    ierr  = MatDestroy(&colGmat);CHKERRQ(ierr); /* we don't need the flatted adjacency any more */
    ierr  = ISGetLocalSize(dofPerm,&nLocal);CHKERRQ(ierr);
    ierr  = PetscCDCreate(nLocal,&agg3_local);CHKERRQ(ierr); /* first calculate localized aggs, then convert to globalized aggs */
  }
  ierr = PCGAMGAggSectionGetFullAggs(pc,sub_agg_lists,columnsLocal,agg3_local,&dofGmat,&coarseColumns,dofPerm);CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)columns,"DofColumns_coarse",(PetscObject)coarseColumns);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&coarseColumns);CHKERRQ(ierr);
  *Gmat = dofGmat;
  { /* convert agg3_local to agg3 */
    PetscInt vStart, vEnd;
    PetscInt vStartOld, vEndOld;
    PetscInt a;
    const PetscInt *dofVal;

    ierr = ISGetIndices(dofPerm,&dofVal);CHKERRQ(ierr);
    ierr = PetscSectionGetOffsetRange(columnsLocal,&vStart,&vEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetOffsetRange(columns,&vStartOld,&vEndOld);CHKERRQ(ierr);
    for (a = 0; a < agg3_local->size; a++) {
      PetscInt size, thisHead;
      PetscCDPos pos;

      ierr = PetscCDSizeAt(agg3_local,a,&size);CHKERRQ(ierr);
      if (!size) continue;
      ierr = PetscCDGetHeadPos(agg3_local,a,&pos);CHKERRQ(ierr);
      thisHead = -1;
      while (pos) {
        PetscInt lid, gid;

        ierr = PetscLLNGetID(pos, &lid);CHKERRQ(ierr);
        gid  = dofVal[lid - vStart];
        if (gid >= vStartOld) {
          thisHead = gid - vStartOld;
          break;
        }
        ierr = PetscCDGetNextPos(agg3_local,a,&pos);CHKERRQ(ierr);
      }
      if (thisHead == -1) SETERRQ1(PETSC_COMM_SELF,PETSC_ERR_PLIB,"Could not find local dof for aggregate %d",a);
      ierr = PetscCDGetHeadPos(agg3_local,a,&pos);CHKERRQ(ierr);
      while (pos) {
        PetscInt lid, gid;

        ierr = PetscLLNGetID(pos, &lid);CHKERRQ(ierr);
        gid  = dofVal[lid - vStart];
        ierr = PetscCDAppendID(agg3,thisHead,gid);CHKERRQ(ierr);
        ierr = PetscCDGetNextPos(agg3_local,a,&pos);CHKERRQ(ierr);
      }
    }
    ierr = ISRestoreIndices(dofPerm,&dofVal);CHKERRQ(ierr);
  }
  ierr = PetscCDDestroy(sub_agg_lists);CHKERRQ(ierr);
  ierr = PetscCDDestroy(agg3_local);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&columnsLocal);CHKERRQ(ierr);
  ierr = ISDestroy(&dofPerm);CHKERRQ(ierr);
  *agg_lists = agg3;
  PetscFunctionReturn(0);
}

/* Simply call the prolongator routine from the columnPC */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGProlongator_DofCol"
PetscErrorCode PCGAMGProlongator_DofCol(PC pc,const Mat Amat,const Mat Gmat,PetscCoarsenData *agg_lists,Mat *P)
{
  PC_GAMG_DofCol *dofcol;
  PC_GAMG        *subgamg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);
  PetscValidHeaderSpecific(Gmat,MAT_CLASSID,3);
  ierr    = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  subgamg = PCGAMGDofColGetSubGAMG(dofcol);
  ierr    = PCGAMGDofColGiveSubGAMGData(pc);
  ierr    = subgamg->ops->prolongator(dofcol->columnPC,Amat,Gmat,agg_lists,P);CHKERRQ(ierr);
  ierr    = PCGAMGDofColTakeSubGAMGData(pc);
  PetscFunctionReturn(0);
}

/* Simply call the optprol routine from the columnPC */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGOptprol_DofCol"
PetscErrorCode PCGAMGOptprol_DofCol(PC pc,const Mat Amat,Mat *P)
{
  PC_GAMG_DofCol *dofcol;
  PC_GAMG        *subgamg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);
  ierr    = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  subgamg = PCGAMGDofColGetSubGAMG(dofcol);
  ierr    = PCGAMGDofColGiveSubGAMGData(pc);
  ierr    = subgamg->ops->optprol(dofcol->columnPC,Amat,P);CHKERRQ(ierr);
  ierr    = PCGAMGDofColTakeSubGAMGData(pc);
  PetscFunctionReturn(0);
}

/* Simply call the default data routine from the columnPC */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGCreateDefaultData_DofCol"
PetscErrorCode PCGAMGCreateDefaultData_DofCol(PC pc,const Mat Amat)
{
  PC_GAMG_DofCol *dofcol;
  PC_GAMG        *subgamg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(Amat,MAT_CLASSID,2);
  ierr    = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  subgamg = PCGAMGDofColGetSubGAMG(dofcol);
  ierr    = PCGAMGDofColGiveSubGAMGData(pc);
  ierr    = subgamg->ops->createdefaultdata(dofcol->columnPC,Amat);CHKERRQ(ierr);
  ierr    = PCGAMGDofColTakeSubGAMGData(pc);
  PetscFunctionReturn(0);
}

/* Call the create level routine from the columnPC, use the reported
 * permutation to propagate the column section to the coarse matrix */
#undef __FUNCT__
#define __FUNCT__ "PCGAMGCreateLevel_DofCol"
PetscErrorCode PCGAMGCreateLevel_DofCol(PC pc,Mat Amat_fine,PetscInt cr_bs,Mat *a_P_inout,Mat *a_Mat_crs,PetscMPIInt *a_nactive_proc,IS *Pcolumnperm)
{
  PetscObject  columns_obj, coarseColumns_obj;
  PetscSection columns, coarseColumns;
  PC_GAMG_DofCol *dofcol;
  PC_GAMG        *subgamg;
  IS             perm;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidHeaderSpecific(Amat_fine,MAT_CLASSID,2);
  ierr = PetscObjectQuery((PetscObject)Amat_fine,"DofColumns",&columns_obj);CHKERRQ(ierr);
  if (!columns_obj) SETERRQ(PetscObjectComm((PetscObject)Amat_fine),PETSC_ERR_ARG_WRONGSTATE,"Could not get the dof columns from the adjacency matrix");
  columns = (PetscSection) columns_obj;
  ierr = PetscObjectQuery((PetscObject)columns,"DofColumns_coarse",&coarseColumns_obj);CHKERRQ(ierr);
  if (!coarseColumns_obj) SETERRQ(PetscObjectComm((PetscObject)columns),PETSC_ERR_ARG_WRONGSTATE,"Could not get the coarse dof columns");
  coarseColumns = (PetscSection) coarseColumns_obj;
  ierr    = PetscObjectReference((PetscObject)coarseColumns);CHKERRQ(ierr);
  ierr    = PetscObjectCompose((PetscObject)columns,"DofColumns_coarse",NULL);CHKERRQ(ierr);
  ierr    = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  subgamg = PCGAMGDofColGetSubGAMG(dofcol);
  ierr    = PCGAMGDofColGiveSubGAMGData(pc);
  ierr    = subgamg->ops->createlevel(dofcol->columnPC,Amat_fine,cr_bs,a_P_inout,a_Mat_crs,a_nactive_proc,&perm);CHKERRQ(ierr);
  ierr    = PCGAMGDofColTakeSubGAMGData(pc);
  if (Pcolumnperm) {
    ierr = PetscObjectReference((PetscObject)perm);CHKERRQ(ierr);
    *Pcolumnperm = perm;
  }
  if (perm) {
    PetscLayout ccLayout;
    const PetscInt *ids;
    PetscInt bs, pStart, pEnd, vStart, vEnd, p, v;
    PetscSF permSF;
    PetscInt nloc;
    PetscInt *dof2col, *dof2colperm, ncol, coloff, thiscol, coldof, voff;
    PetscSection permSec;

    ierr = ISGetBlockSize(perm,&bs);CHKERRQ(ierr);

    if (bs > 1) {
      PetscInt nlocbs, i;
      IS permbs1;
      PetscInt *idsbs1;

      ierr = ISGetLocalSize(perm,&nlocbs);CHKERRQ(ierr);
      ierr = ISGetIndices(perm,&ids);CHKERRQ(ierr);
      ierr = PetscMalloc1(nlocbs / bs,&idsbs1);CHKERRQ(ierr);
      for (i = 0; i < nlocbs / bs; i++) {
        idsbs1[i] = ids[i*bs]/bs;
      }
      ierr = ISRestoreIndices(perm,&ids);CHKERRQ(ierr);
      ierr = ISCreateGeneral(PetscObjectComm((PetscObject)perm),nlocbs / bs,idsbs1,PETSC_OWN_POINTER,&permbs1);CHKERRQ(ierr);
      ierr = ISDestroy(&perm);CHKERRQ(ierr);
      perm = permbs1;
    }

    ierr = ISGetLocalSize(perm,&nloc);CHKERRQ(ierr);
    ierr = ISGetIndices(perm,&ids);CHKERRQ(ierr);
    ierr = PetscLayoutCreate(PetscObjectComm((PetscObject)coarseColumns),&ccLayout);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(coarseColumns,&pStart,&pEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetOffsetRange(coarseColumns,&vStart,&vEnd);CHKERRQ(ierr);
    ierr = PetscLayoutSetLocalSize(ccLayout,vEnd-vStart);CHKERRQ(ierr);
    ierr = PetscLayoutSetUp(ccLayout);CHKERRQ(ierr);
    ierr = PetscSFCreate(PetscObjectComm((PetscObject)perm),&permSF);CHKERRQ(ierr);
    ierr = PetscSFSetGraphLayout(permSF,ccLayout,nloc,NULL,PETSC_OWN_POINTER,ids);CHKERRQ(ierr);
    ierr = PetscMalloc2(vEnd-vStart,&dof2col,nloc,&dof2colperm);CHKERRQ(ierr);
    for (p = pStart; p < pEnd; p++) {
      PetscInt dof, off, v;
      ierr = PetscSectionGetDof(coarseColumns,p,&dof);CHKERRQ(ierr);
      ierr = PetscSectionGetOffset(coarseColumns,p,&off);CHKERRQ(ierr);
      for (v = off; v < off+dof; v++) {
        dof2col[off - vStart] = p;
      }
    }
    ierr = PetscSFBcastBegin(permSF,MPIU_INT,dof2col,dof2colperm);CHKERRQ(ierr);
    ierr = PetscSFBcastEnd(permSF,MPIU_INT,dof2col,dof2colperm);CHKERRQ(ierr);
    ierr = ISRestoreIndices(perm,&ids);CHKERRQ(ierr);
    ierr = PetscSFDestroy(&permSF);CHKERRQ(ierr);
    ierr = PetscLayoutDestroy(&ccLayout);CHKERRQ(ierr);
    ierr = PetscSectionCreate(PetscObjectComm((PetscObject)coarseColumns),&permSec);CHKERRQ(ierr);
    ncol = 0;
    for (v = 0; v < nloc; v++) {
      if (v == nloc - 1 || dof2colperm[v] != dof2colperm[v+1]) ncol++;
    }
    ierr = MPI_Scan(&ncol,&coloff,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)coarseColumns));CHKERRQ(ierr);
    coloff -= ncol;
    ierr = PetscSectionSetChart(permSec,coloff,coloff+ncol);CHKERRQ(ierr);
    for (v = 0, thiscol = 0, coldof = 0; v < nloc; v++) {
      coldof++;
      if (v == nloc - 1 || dof2colperm[v] != dof2colperm[v+1]) {
        ierr = PetscSectionSetDof(permSec,(thiscol++) + coloff,coldof);CHKERRQ(ierr);
        coldof = 0;
      }
    }
    ierr = PetscSectionSetUp(permSec);CHKERRQ(ierr);
    ierr = MPI_Scan(&nloc,&voff,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)permSec));CHKERRQ(ierr);
    voff -= nloc;
    for (p = coloff; p < coloff + ncol; p++) {
      PetscInt off;

      ierr = PetscSectionGetOffset(permSec,p,&off);CHKERRQ(ierr);
      ierr = PetscSectionSetOffset(permSec,p,off+voff);CHKERRQ(ierr);
    }
    ierr = PetscSectionDestroy(&coarseColumns);CHKERRQ(ierr);
    ierr = PetscFree2(dof2col,dof2colperm);CHKERRQ(ierr);
    coarseColumns = permSec;
  }
  ierr = PetscObjectCompose((PetscObject)(*a_Mat_crs),"DofColumns",(PetscObject)coarseColumns);CHKERRQ(ierr);
  ierr = PetscSectionDestroy(&coarseColumns);CHKERRQ(ierr);
  ierr = ISDestroy(&perm);CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGDofColGetAggregateHeight"
PetscErrorCode PCGAMGDofColGetAggregateHeight(PC pc,PetscInt * aggHeight)
{
  PC_GAMG_DofCol *dofcol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  PetscValidIntPointer(aggHeight,2);
  ierr = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  *aggHeight = dofcol->aggHeight;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGDofColSetAggregateHeight"
PetscErrorCode PCGAMGDofColSetAggregateHeight(PC pc,PetscInt aggHeight)
{
  PC_GAMG_DofCol *dofcol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  dofcol->aggHeight = aggHeight;
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGDofColGetColumnGAMGType"
PetscErrorCode PCGAMGDofColGetColumnGAMGType(PC pc,PCGAMGType *type)
{
  PC_GAMG_DofCol *dofcol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  ierr = PCGAMGGetType(dofcol->columnPC,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGDofColSetColumnGAMGType"
PetscErrorCode PCGAMGDofColSetColumnGAMGType(PC pc,PCGAMGType type)
{
  PC_GAMG_DofCol *dofcol;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  ierr = PCGAMGSetType(dofcol->columnPC,type);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGSetFromOptions_DofCol"
PetscErrorCode PCGAMGSetFromOptions_DofCol(PetscOptions *PetscOptionsObject,PC pc)
{
  const char     *prefix;
  PC_GAMG_DofCol *dofcol;
  const char     *type;
  char           newtype[256];
  PetscBool      flg;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  ierr = PCGAMGDofColGetColumnGAMGType(pc,&type);CHKERRQ(ierr);
  ierr = PetscOptionsHead(PetscOptionsObject,"GAMG DofCol");CHKERRQ(ierr);
  {
    ierr = PetscOptionsInt("-pc_gamg_dofcol_aggregate_height","Desired height of aggregates within columns","PCGAMGDofColSetAggregateHeight",dofcol->aggHeight,&dofcol->aggHeight,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsString("-pc_gamg_dofcol_column_gamg_type","Type of GAMG to use for columns","PCGAMGDofColSetColumnGAMGType",type,newtype,256,&flg);CHKERRQ(ierr);
  }
  ierr = PetscOptionsTail();CHKERRQ(ierr);
  if (flg) {
    ierr = PCGAMGDofColSetColumnGAMGType(pc,newtype);CHKERRQ(ierr);
  }
  ierr = PetscObjectGetOptionsPrefix((PetscObject)pc,&prefix);CHKERRQ(ierr);
  ierr = PetscObjectSetOptionsPrefix((PetscObject)dofcol->columnPC,prefix);CHKERRQ(ierr);
  ierr = PetscObjectAppendOptionsPrefix((PetscObject)dofcol->columnPC,"columns_");CHKERRQ(ierr);
  ierr = PCSetFromOptions(dofcol->columnPC);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGDestroy_DofCol"
PetscErrorCode PCGAMGDestroy_DofCol(PC pc)
{
  PC_GAMG_DofCol *dofcol;
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  PetscValidHeaderSpecific(pc,PC_CLASSID,1);
  ierr = PCGAMGGetDofCol(pc,&dofcol);CHKERRQ(ierr);
  ierr = PCDestroy(&(dofcol->columnPC));CHKERRQ(ierr);
  ierr = PetscFree(pc_gamg->subctx);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGCreate_DofCol"
PetscErrorCode PCGAMGCreate_DofCol(PC pc)
{
  PetscErrorCode ierr;
  PC_MG          *mg      = (PC_MG*)pc->data;
  PC_GAMG        *pc_gamg = (PC_GAMG*)mg->innerctx;
  PC_GAMG_DofCol *dofcol;

  PetscFunctionBeginUser;
  /* create sub context for SA */
  ierr            = PetscNewLog(pc,&dofcol);CHKERRQ(ierr);
  pc_gamg->subctx = dofcol;
  dofcol->aggHeight = 3;
  dofcol->useLinear = PETSC_TRUE;
  ierr = PCCreate(PetscObjectComm((PetscObject)pc),&(dofcol->columnPC));CHKERRQ(ierr);
  ierr = PCSetType(dofcol->columnPC,PCGAMG);CHKERRQ(ierr);
  ierr = PetscObjectIncrementTabLevel((PetscObject)(dofcol->columnPC),(PetscObject)pc,1);CHKERRQ(ierr);

  pc_gamg->ops->setfromoptions = PCGAMGSetFromOptions_DofCol;
  pc_gamg->ops->destroy        = PCGAMGDestroy_DofCol;
  /* reset does not do anything; setup not virtual */

  /* set internal function pointers */
  pc_gamg->ops->graph       = PCGAMGGraph_DofCol;
  pc_gamg->ops->coarsen     = PCGAMGCoarsen_DofCol;
  pc_gamg->ops->prolongator = PCGAMGProlongator_DofCol;
  pc_gamg->ops->optprol     = PCGAMGOptprol_DofCol;
  pc_gamg->ops->createdefaultdata = PCGAMGCreateDefaultData_DofCol;
  pc_gamg->ops->createlevel = PCGAMGCreateLevel_DofCol;

  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "PCGAMGRegister_DofCol"
static PetscErrorCode PCGAMGRegister_DofCol (void)
{
  PetscErrorCode ierr;
  PetscFunctionBeginUser;
  ierr = DofColumnsInitializePackage();CHKERRQ(ierr);
  ierr = PCGAMGRegister(PCGAMGDOFCOL, PCGAMGCreate_DofCol);CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatSetDofColumns"
PetscErrorCode MatSetDofColumns(Mat A, PetscSection columns)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DofColumnsInitializePackage();CHKERRQ(ierr);
  ierr = PetscObjectCompose((PetscObject)A,"DofColumns",(PetscObject)columns);CHKERRQ(ierr);
#if defined(PETSC_USE_DEBUG)
  {
    PetscLayout layout;
    PetscInt    pStart, pEnd, vStart, vEnd, n, off;
    PetscInt    bs;

    ierr = MatGetLayouts(A,&layout,NULL);CHKERRQ(ierr);
    ierr = MatGetBlockSize(A,&bs);CHKERRQ(ierr);
    ierr = PetscSectionGetChart(columns,&pStart,&pEnd);CHKERRQ(ierr);
    ierr = PetscSectionGetOffsetRange(columns,&vStart,&vEnd);CHKERRQ(ierr);
    if ((pEnd > pStart) && (vStart != layout->rstart/bs || vEnd != layout->rend/bs)) {
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Expected the column section offset range to be [%d, %d), got [%d, %d)",layout->rstart/bs,layout->rend/bs,vStart,vEnd);
    }
    n    = pEnd - pStart;
    ierr = MPI_Scan(&n,&off,1,MPIU_INT,MPI_SUM,PetscObjectComm((PetscObject)columns));CHKERRQ(ierr);
    off -= n;
    if ((pEnd > pStart) && (off != pStart)) {
      SETERRQ4(PETSC_COMM_SELF,PETSC_ERR_ARG_SIZ,"Expected the column section chart to be [%d, %d), got [%d, %d)",off,off+n,pStart,pEnd);
    }
  }
#endif
  PetscFunctionReturn(0);
}

#undef __FUNCT__
#define __FUNCT__ "MatGetDofColumns"
PetscErrorCode MatGetDofColumns(Mat A, PetscSection *columns)
{
  PetscObject    obj;
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DofColumnsInitializePackage();CHKERRQ(ierr);
  ierr = PetscObjectQuery((PetscObject)A,"DofColumns",&obj);CHKERRQ(ierr);
  *columns = (PetscSection) obj;
  PetscFunctionReturn(0);
}

static PetscBool DofColumnsInitialized;

/* Either call this function from your code, or load the shared library
 * dynamically by adding "-dll_append /path/to/libdofcolumns.so" to your
 * command line argument */
#undef __FUNCT__
#define __FUNCT__ "DofColumnsInitializePackage"
PetscErrorCode DofColumnsInitializePackage(void)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (DofColumnsInitialized) PetscFunctionReturn(0);
  DofColumnsInitialized = PETSC_TRUE;
  ierr = PCGAMGRegister_DofCol();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

#if defined(PETSC_HAVE_DYNAMIC_LIBRARIES)
#undef __FUNCT__
#define __FUNCT__ "PetscDLLibraryRegister_dofcolumns"
PETSC_EXTERN PetscErrorCode PetscDLLibraryRegister_dofcolumns(void)
{
  PetscErrorCode ierr;

  PetscFunctionBeginUser;
  ierr = DofColumnsInitializePackage();CHKERRQ(ierr);
  PetscFunctionReturn(0);
}
#endif

