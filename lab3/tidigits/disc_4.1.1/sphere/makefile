##########################################
# NIST Speech Header Resources (SPHERE)  #
# Release 1.5 (beta)                     #
# Stan Janet (stan@jaguar.ncsl.nist.gov) #
# October 1990                           #
##########################################

V		= 1.5
# current release id

CFLAGS		=

BINDIR		= ./bin

MOD		= h_modify.o
MODC		= h_modify.c

UTIL		= sp_utils.o spinput.o spoutput.o stdfield.o spmalloc.o
UTILC		= sp_utils.c spinput.c spoutput.c stdfield.c spmalloc.c

TOOL		= h_read   h_strip   h_add   h_test   \
			h_delete   h_edit   h_nlrm
TOOLC		= h_read.c h_strip.c h_add.c h_test.c \
			h_delete.c h_edit.c h_nlrm.c

LOCAL		=
LOCALC		=

I		= header.h sp.h

BAKDIR		= Old
C_HEADER	= header.c
SH_HEADER	= header.sh
RDIR		= sphere-$(V)
TARFILE		= $(RDIR).tar

LIBNAME		= sp
LINTLIB		= llib-l$(LIBNAME).ln
LIB		= lib$(LIBNAME).a

MAKEFILE	= makefile.
MANPAGES	= sphere.3 h_read.1
DOC		= readme.doc changes.doc disclaim.doc grammar.doc
MISC		= $(MANPAGES) $(DOC)
HFILES		= header.h sp.h sysfunc.h sysparam.h version.h
CFILES		= $(UTILC) $(TOOLC) $(MODC)
SCRIPTS		=




all: $(LIB) $(LINTLIB) $(MOD) programs

install:
	test ! -d $(BINDIR) && mkdir $(BINDIR)
	mv $(TOOL) $(LOCAL) $(SCRIPTS) $(BINDIR)

.c.o:
	cc $(CFLAGS) -c $*.c


backup:
	test ! -d $(BAKDIR) || rm -rf $(BAKDIR)
	mkdir $(BAKDIR)
	cp $(CFILES) $(HFILES) $(SCRIPTS) $(MISC) $(BAKDIR)


release:
	@test ! -d $(RDIR) || rm -rf $(RDIR)
	@mkdir $(RDIR)
	for F in $(CFILES) $(HFILES) ; do \
		sed "s/-V-/$(V)/" < $(C_HEADER) | cat - $$F > $(RDIR)/$$F; \
	done
	for F in $(SCRIPTS) $(MAKEFILE) ; do \
		sed "s/-V-/$(V)/" < $(SH_HEADER) | \
		cat - $$F | \
		sed	-e 's/^CFLAGS\([	]*\)=.*$$/CFLAGS\1=/' \
			-e 's/^LOCAL\([^=]*\)=.*$$/LOCAL\1=/' \
		> $(RDIR)/$$F; \
	done
	for F in $(MANPAGES) ; do \
		nroff -man $$F > tmp_$$$$ ; \
		OUT=`echo $$F | sed 's/\.[0-9]$$/.doc/'` ; \
		troff2ascii < tmp_$$$$ > $(RDIR)/$${OUT} ; \
		rm tmp_$$$$ ; \
	done
	cp $(MISC) $(RDIR)
	sed "s/-V-/$(V)/" < version.h  > $(RDIR)/version.h
	sed "s/-V-/$(V)/" < readme.doc > $(RDIR)/readme.doc
	-chmod 644 $(RDIR)/*
	-chmod +x $(RDIR)/$(SCRIPTS)
	tar cf $(TARFILE) $(RDIR)
	compress -vf $(TARFILE)


programs: $(TOOL) $(LOCAL)

# Tools in standard distribution
#
h_strip: h_strip.o $(LIB) $I
	cc $(CFLAGS) -o $@ h_strip.o $(LIB)

h_add: h_add.o $(LIB) $I
	cc $(CFLAGS) -o $@ h_add.o $(LIB)

h_read: h_read.o $(LIB) $I
	cc $(CFLAGS) -o $@ h_read.o $(LIB)

h_test: h_test.o $(LIB) $I
	cc $(CFLAGS) -o $@ h_test.o $(LIB)

h_delete: h_delete.o $(MOD) $(LIB) $I
	cc $(CFLAGS) -o $@ h_delete.o $(MOD) $(LIB)

h_edit: h_edit.o $(MOD) $(LIB) $I
	cc $(CFLAGS) -o $@ h_edit.o $(MOD) $(LIB)

h_nlrm: h_nlrm.o $(MOD) $(LIB) $I
	cc $(CFLAGS) -o $@ h_nlrm.o $(MOD) $(LIB)


# Local tools, not in DARPA distribution
#
h_conv2: h_conv2.o $(LIB) $I
	cc $(CFLAGS) -o $@ h_conv2.o $(LIB)

ti2wav: ti2wav.o $(LIB) $I
	cc $(CFLAGS) -o $@ ti2wav.o $(LIB)

# on systems without "ranlib", comment out the "ranlib" line
# or ignore the error message that "make" will (likely) generate
# when sh's attempt at executing the line fails
$(LIB): $(UTIL)
	ar rc $(LIB) $(UTIL)
	-ranlib $(LIB)


$(UTIL): $I


$(LINTLIB): $(UTILC)
	lint -C$(LIBNAME) $(UTILC)


lintr: $(LINTLIB)
	lint h_read.c $(LINTLIB)

lints: $(LINTLIB)
	lint h_strip.c $(LINTLIB)

lintt: $(LINTLIB)
	lint h_test.c $(LINTLIB)

linta: $(LINTLIB)
	lint h_add.c $(LINTLIB)

lint: linta lints lintr lintt
