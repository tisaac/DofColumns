static char help[] = "3D, bi-linear quadrilateral (Q1), displacement finite element formulation\n\
of linear elasticity, with anisotropy in the z-direction and a Robin-type lateral boundary condition.\n\
Unit square domain with Dirichlet boundary condition on the z=0 side in the z-direction only only.\n\
Load of 1.0 in x + 2y direction on all nodes (not a true uniform load).\n\n";

#include <petscksp.h>
#include <petscfe.h>
#include <DofColumns.h>

static PetscBool log_stages = PETSC_TRUE;
static PetscErrorCode MaybeLogStagePush(PetscLogStage stage) { return log_stages ? PetscLogStagePush(stage) : 0; }
static PetscErrorCode MaybeLogStagePop() { return log_stages ? PetscLogStagePop() : 0; }

#undef __FUNCT__
#define __FUNCT__ "main"
int main(int argc,char **args)
{
  Mat            Amat;
  PetscErrorCode ierr;
  PetscInt       m,nn,M,Istart,Iend,i,j,k,ii,jj,kk,ic,ne=4,id;
  PetscReal      x,y,z,h,*coords,soft_alpha=1.e0,epsilon=1.0;
  PetscBool      two_solves=PETSC_FALSE,test_nonzero_cols=PETSC_FALSE,use_nearnullspace=PETSC_FALSE;
  Vec            xx,bb;
  KSP            ksp;
  MPI_Comm       comm;
  PetscMPIInt    npe,mype;
  PC             pc;
  PetscScalar    DD[24][24],DD2[24][24];
  PetscLogStage  stage[6];
  PetscScalar    DD1[24][24];
  PetscReal      nu = 0.25, E = 1.0, beta = 1.0;
  PCType         type;

  PetscInitialize(&argc,&args,(char*)0,help);
  comm = PETSC_COMM_WORLD;
  ierr  = MPI_Comm_rank(comm, &mype);CHKERRQ(ierr);
  ierr  = MPI_Comm_size(comm, &npe);CHKERRQ(ierr);
  ierr  = DofColumnsInitializePackage();CHKERRQ(ierr);

  ierr = PetscOptionsBegin(comm,NULL,"3D bilinear Q1 elasticity options","");CHKERRQ(ierr);
  {
    char nestring[256];
    ierr = PetscSNPrintf(nestring,sizeof nestring,"number of elements in each direction, ne+1 must be a multiple of %D (sizes^{1/3})",(PetscInt)(PetscPowReal((PetscReal)npe,1./3.) + .5));CHKERRQ(ierr);
    ierr = PetscOptionsInt("-ne",nestring,"",ne,&ne,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-log_stages","Log stages of solve separately","",log_stages,&log_stages,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-alpha","material coefficient inside circle","",soft_alpha,&soft_alpha,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-beta","Robin boundary condition coefficient","",beta,&beta,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-E","Young's modulus","",E,&E,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-nu","Poisson's ratio","",nu,&nu,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-epsilon","scale of z coordinates","",epsilon,&epsilon,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsReal("-epsilon","scale of z coordinates","",epsilon,&epsilon,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-two_solves","solve additional variant of the problem","",two_solves,&two_solves,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-test_nonzero_cols","nonzero test","",test_nonzero_cols,&test_nonzero_cols,NULL);CHKERRQ(ierr);
    ierr = PetscOptionsBool("-use_mat_nearnullspace","MatNearNullSpace API test","",use_nearnullspace,&use_nearnullspace,NULL);CHKERRQ(ierr);
  }
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  if (log_stages) {
    ierr = PetscLogStageRegister("Setup", &stage[0]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("Solve", &stage[1]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("2nd Setup", &stage[2]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("2nd Solve", &stage[3]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("3rd Setup", &stage[4]);CHKERRQ(ierr);
    ierr = PetscLogStageRegister("3rd Solve", &stage[5]);CHKERRQ(ierr);
  } else {
    for (i=0; i<(PetscInt)(sizeof(stage)/sizeof(stage[0])); i++) stage[i] = -1;
  }

  h = 1./ne; nn = ne+1;
  /* ne*ne; number of global elements */
  M = 3*nn*nn*nn; /* global number of equations */
  if (npe==2) {
    if (mype==1) m=0;
    else m = nn*nn*nn;
    npe = 1;
  } else {
    m = nn*nn*nn/npe;
    if (mype==npe-1) m = nn*nn*nn - (npe-1)*m;
  }
  m *= 3; /* number of equations local*/
  /* Setup solver, get PC type and pc */
  ierr = KSPCreate(PETSC_COMM_WORLD,&ksp);CHKERRQ(ierr);
  ierr = KSPSetType(ksp, KSPCG);CHKERRQ(ierr);
  ierr = KSPSetComputeSingularValues(ksp, PETSC_TRUE);CHKERRQ(ierr);
  ierr = KSPGetPC(ksp, &pc);CHKERRQ(ierr);
  ierr = PCSetType(pc, PCGAMG);CHKERRQ(ierr); /* default */
  ierr = KSPSetFromOptions(ksp);CHKERRQ(ierr);
  ierr = PCGetType(pc, &type);CHKERRQ(ierr);

  {
    /* configuration */
    const PetscInt NP = (PetscInt)(PetscPowReal((PetscReal)npe,1./2.) + .5);
    const PetscInt ipx = mype%NP, ipy = mype/NP;
    const PetscInt Ni0 = ipx*(nn/NP), Nj0 = ipy*(nn/NP), Nk0 = 0;
    const PetscInt Ni1 = Ni0 + (m>0 ? (nn/NP) : 0), Nj1 = Nj0 + (nn/NP), Nk1 = nn;
    const PetscInt NN  = nn/NP, id0 = nn * (ipy*nn*NN + ipx*NN*NN);
    PetscInt       *d_nnz, *o_nnz,osz[4]={0,9,15,19},nbc;
    PetscScalar    vv[24], v2[24];
    if (npe!=NP*NP) SETERRQ1(comm,PETSC_ERR_ARG_WRONG, "npe=%d: npe^{1/2} must be integer",npe);
    if (nn!=NP*(nn/NP)) SETERRQ1(comm,PETSC_ERR_ARG_WRONG, "-ne %d: (ne+1)%(npe^{1/2}) must equal zero",ne);

    /* count nnz */
    ierr = PetscMalloc1(m+1, &d_nnz);CHKERRQ(ierr);
    ierr = PetscMalloc1(m+1, &o_nnz);CHKERRQ(ierr);
    for (j=Nj0,ic=0; j<Nj1; j++) {
      for (i=Ni0; i<Ni1; i++) {
        for (k=Nk0; k<Nk1; k++) {
          nbc = 0;
          if (i==Ni0 || i==Ni1-1) nbc++;
          if (j==Nj0 || j==Nj1-1) nbc++;
          if (k==Nk0 || k==Nk1-1) nbc++;
          for (jj=0; jj<3; jj++,ic++) {
            d_nnz[ic] = 3*(27-osz[nbc]);
            o_nnz[ic] = 3*osz[nbc];
          }
        }
      }
    }
    if (ic != m) SETERRQ2(PETSC_COMM_SELF,PETSC_ERR_PLIB,"ic %D does not equal m %D",ic,m);

    /* create stiffness matrix */
    ierr = MatCreate(comm,&Amat);CHKERRQ(ierr);
    ierr = MatSetSizes(Amat,m,m,M,M);CHKERRQ(ierr);
    ierr = MatSetBlockSize(Amat,3);CHKERRQ(ierr);
    ierr = MatSetType(Amat,MATAIJ);CHKERRQ(ierr);
    ierr = MatSeqAIJSetPreallocation(Amat,0,d_nnz);CHKERRQ(ierr);
    ierr = MatMPIAIJSetPreallocation(Amat,0,d_nnz,0,o_nnz);CHKERRQ(ierr);

    ierr = PetscFree(d_nnz);CHKERRQ(ierr);
    ierr = PetscFree(o_nnz);CHKERRQ(ierr);

    ierr = MatGetOwnershipRange(Amat,&Istart,&Iend);CHKERRQ(ierr);

    if (m != Iend - Istart) SETERRQ3(PETSC_COMM_SELF,PETSC_ERR_PLIB,"m %D does not equal Iend %D - Istart %D",m,Iend,Istart);
    /* Generate vectors */
    ierr = VecCreate(comm,&xx);CHKERRQ(ierr);
    ierr = VecSetSizes(xx,m,M);CHKERRQ(ierr);
    ierr = VecSetBlockSize(xx,3);CHKERRQ(ierr);
    ierr = VecSetFromOptions(xx);CHKERRQ(ierr);
    ierr = VecDuplicate(xx,&bb);CHKERRQ(ierr);
    ierr = VecSet(bb,.0);CHKERRQ(ierr);
    /* generate element matrices */
    {
      {
        PetscSpace      P;
        DM              K;
        PetscDualSpace  Q;
        PetscFE         fem;
        PetscReal      *D;
        PetscQuadrature q;

        ierr = PetscSpaceCreate(comm,&P);CHKERRQ(ierr);
        ierr = PetscSpaceSetType(P,PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
        ierr = PetscSpacePolynomialSetTensor(P,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscSpacePolynomialSetNumVariables(P,3);CHKERRQ(ierr);
        ierr = PetscSpaceSetOrder(P,1);CHKERRQ(ierr);
        ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
        /* Create dual space */
        ierr = PetscDualSpaceCreate(comm, &Q);CHKERRQ(ierr);
        ierr = PetscDualSpaceSetType(Q, PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
        ierr = PetscDualSpaceCreateReferenceCell(Q, 3, PETSC_FALSE, &K);CHKERRQ(ierr);
        ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
        ierr = DMDestroy(&K);CHKERRQ(ierr);
        ierr = PetscDualSpaceSetOrder(Q, 1);CHKERRQ(ierr);
        ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
        /* Create element */
        ierr = PetscFECreate(comm, &fem);CHKERRQ(ierr);
        ierr = PetscFESetType(fem,PETSCFEBASIC);CHKERRQ(ierr);
        ierr = PetscFESetBasisSpace(fem, P);CHKERRQ(ierr);
        ierr = PetscFESetDualSpace(fem, Q);CHKERRQ(ierr);
        ierr = PetscFESetNumComponents(fem, 1);CHKERRQ(ierr);
        ierr = PetscFESetUp(fem);CHKERRQ(ierr);
        ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
        ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
        /* Create quadrature (with specified order if given) */
        ierr = PetscDTGaussTensorQuadrature(3, 2, -1.0, 1.0, &q);CHKERRQ(ierr);
        ierr = PetscFESetQuadrature(fem, q);CHKERRQ(ierr);
        ierr = PetscQuadratureDestroy(&q);CHKERRQ(ierr);
        ierr = PetscFEGetDefaultTabulation(fem,NULL,&D,NULL);CHKERRQ(ierr);
        for (i = 0; i < 8; i++) {
          PetscReal swap[3];
          for (j = 0; j < 8; j++) {
            D[24 * i + 3 * j + 0] /= h;
            D[24 * i + 3 * j + 1] /= h;
            D[24 * i + 3 * j + 2] /= (h * epsilon);
          }
          /* swap order: difference between PetscFE numbering and our numbering */
          swap[0] = D[24 * i + 3 * 1 + 0];
          swap[1] = D[24 * i + 3 * 1 + 1];
          swap[2] = D[24 * i + 3 * 1 + 2];
          D[24 * i + 3 * 1 + 0] = D[24 * i + 3 * 3 + 0];
          D[24 * i + 3 * 1 + 1] = D[24 * i + 3 * 3 + 1];
          D[24 * i + 3 * 1 + 2] = D[24 * i + 3 * 3 + 2];
          D[24 * i + 3 * 3 + 0] = swap[0];
          D[24 * i + 3 * 3 + 1] = swap[1];
          D[24 * i + 3 * 3 + 2] = swap[2];
        }
        {
          PetscScalar lambda = E * nu / ((1. + nu) * (1 - 2. * nu));
          PetscScalar mu     = (1./2.) * E / (1. + nu);
          PetscInt i, j, k, u, v, r, s;

          for (i = 0; i < 8; i++) {
            for (j = 0; j < 8; j++) {
              PetscScalar total[9] = {0.};
              for (k = 0; k < 8; k++) {
                for (v = 0; v < 3; v++) {
                  for (u = 0; u < 3; u++) {
                    for (s = 0; s < 3; s++) {
                      for (r = 0; r < 3; r++) {
                        total[v * 3 + u] += h * h * h * epsilon * (lambda * (u == r && v == s) + mu * ((u == v && r == s) + (u == s && v == r))) * D[24 * k + 3 * j + r] * D[24 * k + 3 * i + s];
                      }
                    }
                  }
                }
              }
              for (v = 0; v < 3; v++) {
                for (u = 0; u < 3; u++) {
                  DD1[3 * i + v][3 * j + u] = total[3 * v + u];
                }
              }
            }
          }
        }
        ierr = PetscFEDestroy(&fem);CHKERRQ(ierr);
      }
      {
        PetscSpace      P;
        DM              K;
        PetscDualSpace  Q;
        PetscFE         fem;
        PetscReal      *B;
        PetscQuadrature q;

        ierr = PetscSpaceCreate(comm,&P);CHKERRQ(ierr);
        ierr = PetscSpaceSetType(P,PETSCSPACEPOLYNOMIAL);CHKERRQ(ierr);
        ierr = PetscSpacePolynomialSetTensor(P,PETSC_TRUE);CHKERRQ(ierr);
        ierr = PetscSpacePolynomialSetNumVariables(P,2);CHKERRQ(ierr);
        ierr = PetscSpaceSetOrder(P,1);CHKERRQ(ierr);
        ierr = PetscSpaceSetUp(P);CHKERRQ(ierr);
        /* Create dual space */
        ierr = PetscDualSpaceCreate(comm, &Q);CHKERRQ(ierr);
        ierr = PetscDualSpaceSetType(Q, PETSCDUALSPACELAGRANGE);CHKERRQ(ierr);
        ierr = PetscDualSpaceCreateReferenceCell(Q, 2, PETSC_FALSE, &K);CHKERRQ(ierr);
        ierr = PetscDualSpaceSetDM(Q, K);CHKERRQ(ierr);
        ierr = DMDestroy(&K);CHKERRQ(ierr);
        ierr = PetscDualSpaceSetOrder(Q, 1);CHKERRQ(ierr);
        ierr = PetscDualSpaceSetUp(Q);CHKERRQ(ierr);
        /* Create element */
        ierr = PetscFECreate(comm, &fem);CHKERRQ(ierr);
        ierr = PetscFESetType(fem,PETSCFEBASIC);CHKERRQ(ierr);
        ierr = PetscFESetBasisSpace(fem, P);CHKERRQ(ierr);
        ierr = PetscFESetDualSpace(fem, Q);CHKERRQ(ierr);
        ierr = PetscFESetNumComponents(fem, 1);CHKERRQ(ierr);
        ierr = PetscFESetUp(fem);CHKERRQ(ierr);
        ierr = PetscSpaceDestroy(&P);CHKERRQ(ierr);
        ierr = PetscDualSpaceDestroy(&Q);CHKERRQ(ierr);
        /* Create quadrature (with specified order if given) */
        ierr = PetscDTGaussTensorQuadrature(2, 2, -1.0, 1.0, &q);CHKERRQ(ierr);
        ierr = PetscFESetQuadrature(fem, q);CHKERRQ(ierr);
        ierr = PetscQuadratureDestroy(&q);CHKERRQ(ierr);
        ierr = PetscFEGetDefaultTabulation(fem,&B,NULL,NULL);CHKERRQ(ierr);
        /* BC version of element */
        for (i=0; i<24; i++) {
          for (j=0; j<24; j++) {

            PetscInt ii   = i/3;
            PetscInt idir = i%3;
            PetscInt jj   = j/3;
            PetscInt jdir = j%3;

            if (ii == 0 || ii == 1 || ii == 2 || ii == 3 ||
                jj == 0 || jj == 1 || jj == 2 || jj == 3) {
              if (idir == 2 || jdir == 2) {
                if (i==j) DD2[i][j] = 0.1*DD1[i][j];
                else DD2[i][j] = 0.0;
              }
              else {
                DD2[i][j] = DD1[i][j];
                if (idir == jdir) { /* add the Robin boundary condition */
                  PetscInt k;

                  for (k = 0; k < 4; k++) {
                    DD2[i][j] += h * h * beta * B[4 * k + ii] * B[4 * k + jj];
                  }
                }
              }
            }
            else DD2[i][j] = DD1[i][j];
          }
        }
        /* element residual/load vector */
        for (i=0; i<24; i++) {
          if (i%3==2) vv[i] = h*h;
          else if (i%3==0) vv[i] = 2.0*h*h;
          else vv[i] = .0;
        }
        for (i=0; i<24; i++) {
          PetscInt ii = i/3;
          v2[i] = vv[i];
          if (ii == 0 || ii == 1 || ii == 2 || ii == 3) {
            v2[i] = 0.;
          }
        }
        ierr = PetscFEDestroy(&fem);CHKERRQ(ierr);
      }
    }

    ierr      = PetscMalloc1(m+1, &coords);CHKERRQ(ierr);
    coords[m] = -99.0;

    /* forms the element stiffness and coordinates */
    for (j=Nj0,jj=0,ic=0; j<Nj1; j++,jj++) {
      for (i=Ni0,ii=0; i<Ni1; i++,ii++) {
        for (k=Nk0,kk=0; k<Nk1; k++,kk++,ic++) {

          /* coords */
          x = coords[3*ic] = h*(PetscReal)i;
          y = coords[3*ic+1] = h*(PetscReal)j;
          z = coords[3*ic+2] = h*(PetscReal)k*epsilon;
          /* matrix */
          id = id0 + kk + nn*ii + nn*NN*jj;

          if (i<ne && j<ne && k<ne) {
            /* radius */
            PetscReal radius = PetscSqrtReal((x-.5+h/2)*(x-.5+h/2)+(y-.5+h/2)*(y-.5+h/2)+(z-.5+h/2)*(z-.5+h/2));
            PetscReal alpha = 1.0;
            PetscInt  jx,ix,idx[8] = { id  , id   + nn*1, id   + nn*(NN+1), id+    nn*NN,
                                       id+1, id+1 + nn*1, id+1 + nn*(NN+1), id+1 + nn*NN };

            /* correct indices */
            if (i==Ni1-1 && Ni1!=nn) {
              idx[1] += nn*NN*(NN-1);
              idx[2] += nn*NN*(NN-1);
              idx[5] += nn*NN*(NN-1);
              idx[6] += nn*NN*(NN-1);
            }
            if (j==Nj1-1 && Nj1!=nn) {
              idx[2] += nn*nn*NN-nn*NN*NN;
              idx[3] += nn*nn*NN-nn*NN*NN;
              idx[6] += nn*nn*NN-nn*NN*NN;
              idx[7] += nn*nn*NN-nn*NN*NN;
            }

            if (radius < 0.25) alpha = soft_alpha;

            for (ix=0; ix<24; ix++) {
              for (jx=0;jx<24;jx++) DD[ix][jx] = alpha*DD1[ix][jx];
            }
            if (k>0) {
              ierr = MatSetValuesBlocked(Amat,8,idx,8,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
              ierr = VecSetValuesBlocked(bb,8,idx,(const PetscScalar*)vv,ADD_VALUES);CHKERRQ(ierr);
            } else {
              /* a BC */
              for (ix=0;ix<24;ix++) {
                for (jx=0;jx<24;jx++) DD[ix][jx] = alpha*DD2[ix][jx];
              }
              ierr = MatSetValuesBlocked(Amat,8,idx,8,idx,(const PetscScalar*)DD,ADD_VALUES);CHKERRQ(ierr);
              ierr = VecSetValuesBlocked(bb,8,idx,(const PetscScalar*)v2,ADD_VALUES);CHKERRQ(ierr);
            }
          }
        }
      }

    }
    ierr = MatAssemblyBegin(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = MatAssemblyEnd(Amat,MAT_FINAL_ASSEMBLY);CHKERRQ(ierr);
    ierr = VecAssemblyBegin(bb);CHKERRQ(ierr);
    ierr = VecAssemblyEnd(bb);CHKERRQ(ierr);
    { /* DofColumns: this is where we specify the columns of degrees of freedom */
      PetscSection columns;
      PetscInt p;

      ierr = PetscSectionCreate(comm,&columns);CHKERRQ(ierr);
      ierr = PetscSectionSetChart(columns,id0/nn,id0/nn+NN*NN);CHKERRQ(ierr);
      for (p = id0/nn; p < id0/nn+NN*NN; p++) {
        ierr = PetscSectionSetDof(columns,p,nn);CHKERRQ(ierr);CHKERRQ(ierr);
      }
      ierr = PetscSectionSetUp(columns);CHKERRQ(ierr);
      for (p = id0/nn; p < id0/nn+NN*NN; p++) {
        ierr = PetscSectionSetOffset(columns,p,p*nn);CHKERRQ(ierr);
      }
      ierr = MatSetDofColumns(Amat,columns);CHKERRQ(ierr);
      ierr = PetscSectionDestroy(&columns);CHKERRQ(ierr);
    }
  }

  if (!PETSC_TRUE) {
    PetscViewer viewer;
    ierr = PetscViewerASCIIOpen(comm, "Amat.m", &viewer);CHKERRQ(ierr);
    ierr = PetscViewerSetFormat(viewer, PETSC_VIEWER_ASCII_MATLAB);CHKERRQ(ierr);
    ierr = MatView(Amat,viewer);CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer);
  }

  /* finish KSP/PC setup */
  ierr = KSPSetOperators(ksp, Amat, Amat);CHKERRQ(ierr);
  if (use_nearnullspace) {
    MatNullSpace matnull;
    Vec          vec_coords;
    PetscScalar  *c;

    ierr = VecCreate(MPI_COMM_WORLD,&vec_coords);CHKERRQ(ierr);
    ierr = VecSetBlockSize(vec_coords,3);CHKERRQ(ierr);
    ierr = VecSetSizes(vec_coords,m,PETSC_DECIDE);CHKERRQ(ierr);
    ierr = VecSetUp(vec_coords);CHKERRQ(ierr);
    ierr = VecGetArray(vec_coords,&c);CHKERRQ(ierr);
    for (i=0; i<m; i++) c[i] = coords[i]; /* Copy since Scalar type might be Complex */
    ierr = VecRestoreArray(vec_coords,&c);CHKERRQ(ierr);
    ierr = MatNullSpaceCreateRigidBody(vec_coords,&matnull);CHKERRQ(ierr);
    ierr = MatSetNearNullSpace(Amat,matnull);CHKERRQ(ierr);
    ierr = MatNullSpaceDestroy(&matnull);CHKERRQ(ierr);
    ierr = VecDestroy(&vec_coords);CHKERRQ(ierr);
  } else {
    ierr = PCSetCoordinates(pc, 3, m/3, coords);CHKERRQ(ierr);
  }

  ierr = MaybeLogStagePush(stage[0]);CHKERRQ(ierr);

  /* PC setup basically */
  ierr = KSPSetUp(ksp);CHKERRQ(ierr);

  ierr = MaybeLogStagePop();CHKERRQ(ierr);
  ierr = MaybeLogStagePush(stage[1]);CHKERRQ(ierr);

  /* test BCs */
  if (test_nonzero_cols) {
    VecZeroEntries(xx);
    if (mype==0) VecSetValue(xx,0,1.0,INSERT_VALUES);
    VecAssemblyBegin(xx);
    VecAssemblyEnd(xx);
    KSPSetInitialGuessNonzero(ksp,PETSC_TRUE);
  }

  /* 1st solve */
  ierr = KSPSolve(ksp, bb, xx);CHKERRQ(ierr);

  ierr = MaybeLogStagePop();CHKERRQ(ierr);

  /* 2nd solve */
  if (two_solves) {
    PetscReal emax, emin;

    ierr = MaybeLogStagePush(stage[2]);CHKERRQ(ierr);
    /* PC setup basically */
    ierr = MatScale(Amat, 100000.0);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, Amat, Amat);CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);

    ierr = MaybeLogStagePop();CHKERRQ(ierr);
    ierr = MaybeLogStagePush(stage[3]);CHKERRQ(ierr);
    ierr = KSPSolve(ksp, bb, xx);CHKERRQ(ierr);
    ierr = KSPComputeExtremeSingularValues(ksp, &emax, &emin);CHKERRQ(ierr);

    ierr = MaybeLogStagePop();CHKERRQ(ierr);
    ierr = MaybeLogStagePush(stage[4]);CHKERRQ(ierr);

    /* 3rd solve */
    ierr = MatScale(Amat, 100000.0);CHKERRQ(ierr);
    ierr = KSPSetOperators(ksp, Amat, Amat);CHKERRQ(ierr);
    ierr = KSPSetUp(ksp);CHKERRQ(ierr);

    ierr = MaybeLogStagePop();CHKERRQ(ierr);
    ierr = MaybeLogStagePush(stage[5]);CHKERRQ(ierr);

    ierr = KSPSolve(ksp, bb, xx);CHKERRQ(ierr);

    ierr = MaybeLogStagePop();CHKERRQ(ierr);
  }

  /* Free work space */
  ierr = KSPDestroy(&ksp);CHKERRQ(ierr);
  ierr = VecDestroy(&xx);CHKERRQ(ierr);
  ierr = VecDestroy(&bb);CHKERRQ(ierr);
  ierr = MatDestroy(&Amat);CHKERRQ(ierr);
  ierr = PetscFree(coords);CHKERRQ(ierr);

  ierr = PetscFinalize();
  return 0;
}

