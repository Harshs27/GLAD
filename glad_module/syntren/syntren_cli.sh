#!/bin/sh
#
# Run SynTReN from a jar file
# this is a linux-only version
#-------------------------------------------------------------------------------

#java -Xmx512M -cp SynTReN.jar islab.bayesian.genenetwork.generation.NetworkGeneratorCLI $*
java -Xmx512M -cp SynTReN.jar:lib/xmlpull-1.1.3.1.jar:lib/xpp3_min-1.1.3.4.O.jar:lib/xstream-1.4.11.1.jar:lib/cglib-nodep-2.1_3.jar:lib/colt.jar:lib/commons-math-1.1.jar:lib/xercesImpl.jar:lib/xpp3_min-1.1.3.4.O.jar islab.bayesian.genenetwork.generation.NetworkGeneratorCLI $*
#java -Xmx512M -cp SynTReN.jar:lib/xmlpull-1.1.3.1.jar:lib/xpp3_min-1.1.3.4.O.jar:lib/xstream-1.4.11.1.jar:lib/cglib-nodep-2.1_3.jar:lib/colt.jar:lib/commons-math-1.1.jar:lib/xercesImpl.jar:lib/xpp3_min-1.1.3.4.O.jar expts_gene/syntren/islab.bayesian.genenetwork.generation.NetworkGeneratorCLI $*
