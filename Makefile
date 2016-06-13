DOFCOLUMNS_DIR  := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
DOFCOLUMNS_ARCH := $(if $(PETSC_ARCH),$(PETSC_ARCH),build)

.SECONDEXPANSION:   # to expand $$(@D)/.DIR
.SUFFIXES:          # Clear .SUFFIXES because we don't use implicit rules
.DELETE_ON_ERROR:   # Delete likely-corrupt target file if rule fails

OBJDIR ?= $(DOFCOLUMNS_ARCH)/obj
LIBDIR ?= $(DOFCOLUMNS_ARCH)/lib
INCDIR ?= $(DOFCOLUMNS_ARCH)/include

libdofcolumns.c := DofColumns.c
libdofcolumns.o := $(patsubst %.c,$(OBJDIR)/%.o,$(libdofcolumns.c))
libdofcolumns.h := DofColumns.h

include $(PETSC_DIR)/lib/petsc/conf/variables

# $(call SONAME_FUNCTION,libfoo,abiversion)
SONAME_FUNCTION ?= $(1).$(SL_LINKER_SUFFIX).$(2)
# $(call SL_LINKER_FUNCTION,libfoo,abiversion,libversion)
SL_LINKER_FUNCTION ?= -shared -Wl,-soname,$(call SONAME_FUNCTION,$(notdir $(1)),$(2))

DOFCOLUMNS_VERSION_MAJOR := 0
DOFCOLUMNS_VERSION_MINOR := 1
DOFCOLUMNS_VERSION_PATCH := 0

libdofcolumns_abi_version := $(DOFCOLUMNS_VERSION_MAJOR).$(DOFCOLUMNS_VERSION_MINOR)
libdofcolumns_lib_version := $(libdofcolumns_abi_version).$(DOFCOLUMNS_VERSION_PATCH)
soname_function  = $(call SONAME_FUNCTION,$(1),$(libdofcolumns_abi_version))
libname_function = $(call SONAME_FUNCTION,$(1),$(libdofcolumns_lib_version))
basename_all     = $(basename $(basename $(basename $(basename $(1)))))
sl_linker_args   = $(call SL_LINKER_FUNCTION,$(call basename_all,$@),$(libdofcolumns_abi_version),$(libdofcolumns_lib_version))

libdofcolumns_shared  := $(LIBDIR)/libdofcolumns.$(SL_LINKER_SUFFIX)
libdofcolumns_soname  := $(call soname_function,$(LIBDIR)/libdofcolumns)
libdofcolumns_libname := $(call libname_function,$(LIBDIR)/libdofcolumns)
libdofcolumns_static  := $(LIBDIR)/libdofcolumns.$(AR_LIB_SUFFIX)
libdofcolumns         := $(if $(filter-out no,$(BUILDSHAREDLIB)),$(libdofcolumns_shared) $(libdofcolumns_soname) $(libdofcolumns_libname),$(libdofcolums_static))

all: libdofcolumns

libdofcolumns : $(libdofcolumns)

### Rules

DOFCOLUMNS_LIB_DIR       = $(DOFCOLUMNS_DIR)/$(DOFCOLUMNS_ARCH)/lib
DOFCOLUMNS_C_SH_LIB_PATH = $(CC_LINKER_SLFLAG)$(DOFCOLUMNS_LIB_DIR)
DOFCOLUMNS_LIB           = $(DOFCOLUMNS_C_SH_LIB_PATH) -L$(DOFCOLUMNS_LIB_DIR) -ldofcolumns $(PETSC_KSP_LIB)

# compile an object from a generated c file
$(OBJDIR)/%.o: %.c | $$(@D)/.DIR
	$(PETSC_COMPILE) $(C_DEPFLAGS) $< -o $@

$(libdofcolumns_static): $(libdofcolumns.o) | $$(@D)/.DIR
	$(AR) $(AR_FLAGS) $@ $^
	$(RANLIB) $@

# shared library linking
$(libdofcolumns_libname): $(libdofcolumns.o) | $$(@D)/.DIR
	$(CLINKER) $(sl_linker_args) -o $@ $^ $(UISCE_EXTERNAL_LIB_BASIC)
ifneq ($(DSYMUTIL),true)
	$(DSYMUTIL) $@
endif

$(libdofcolumns_shared): $(libdofcolumns_libname)
	@ln -sf $(notdir $<) $@

$(libdofcolumns_soname): $(libdofcolumns_libname)
	@ln -sf $(notdir $<) $@

# make print VAR=the-variable
print:
	@echo $($(VAR))

# make directories that we need
%/.DIR :
	@mkdir -p $(@D)
	@touch $@

clean:
	rm -rf $(OBJDIR) $(LIBDIR) examples/ex56anisotropic examples/ex56anisotropic.o

install:
	./install.py --prefix=$(DESTDIR)

test:
	@$(PETSC_COMPILE) -I./ examples/ex56anisotropic.c -o examples/ex56anisotropic.o && \
		$(CLINKER) -o examples/ex56anisotropic examples/ex56anisotropic.o $(LDLIBS) $(DOFCOLUMNS_LIB)

MPI_N = 9

test_args = -pc_type gamg -use_mat_nearnullspace -ksp_converged_reason -ksp_rtol 1.e-10 \
						-mg_coarse_ksp_type preonly -mg_levels_ksp_type chebyshev \
						-mg_levels_ksp_chebyshev_esteig \
						-mg_levels_ksp_chebyshev_esteig_random \
						-mg_levels_pc_type bjacobi
run_test  = $(MPIEXEC) -n $(MPI_N) examples/ex56anisotropic $(test_args)

test_sor: test
	$(run_test) -ne 11 -mg_levels_sub_pc_type sor -epsilon 1.0
	$(run_test) -ne 11 -mg_levels_sub_pc_type sor -epsilon 0.1
	$(run_test) -ne 11 -mg_levels_sub_pc_type sor -epsilon 0.01

test_icc: test
	$(run_test) -ne 11 -mg_levels_sub_pc_type icc -epsilon 1.0
	$(run_test) -ne 11 -mg_levels_sub_pc_type icc -epsilon 0.1
	$(run_test) -ne 11 -mg_levels_sub_pc_type icc -epsilon 0.01

test_h: test
	$(run_test) -mg_levels_sub_pc_type icc -epsilon 0.01 -ne 11
	$(run_test) -mg_levels_sub_pc_type icc -epsilon 0.01 -ne 23
	$(run_test) -mg_levels_sub_pc_type icc -epsilon 0.01 -ne 47

test_h_dofcol: test
	$(run_test) -mg_levels_sub_pc_type icc -epsilon 0.01 -ne 11 -pc_gamg_type dofcol
	$(run_test) -mg_levels_sub_pc_type icc -epsilon 0.01 -ne 23 -pc_gamg_type dofcol
	$(run_test) -mg_levels_sub_pc_type icc -epsilon 0.01 -ne 47 -pc_gamg_type dofcol

test_h_weak: test
	$(run_test) -mg_levels_sub_pc_type icc -epsilon 0.01 -ne 8  -beta 0.01
	$(run_test) -mg_levels_sub_pc_type icc -epsilon 0.01 -ne 17 -beta 0.01
	$(run_test) -mg_levels_sub_pc_type icc -epsilon 0.01 -ne 35 -beta 0.01

test_h_weak_dofcol: test
	$(run_test) -mg_levels_sub_pc_type icc -epsilon 0.01 -ne 8  -beta 0.01 -pc_gamg_type dofcol
	$(run_test) -mg_levels_sub_pc_type icc -epsilon 0.01 -ne 17 -beta 0.01 -pc_gamg_type dofcol
	$(run_test) -mg_levels_sub_pc_type icc -epsilon 0.01 -ne 35 -beta 0.01 -pc_gamg_type dofcol

.PHONY: all clean print libdofcolumns install test test_sor test_icc test_h test_h_dofcol

.PRECIOUS: %/.DIR

libdofcolumns.d := $(libdofcolumns.o:%.o=%.d)

$(libdofcolumns.d) : ;

-include $(libdofcolumns.d)
