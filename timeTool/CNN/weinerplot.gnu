#
# Set overall margins for the combined set of plots and size them
# to generate a requested inter-plot spacing
#
#if (!exists("MP_LEFT"))   MP_LEFT = .1
#if (!exists("MP_RIGHT"))  MP_RIGHT = .95
#if (!exists("MP_BOTTOM")) MP_BOTTOM = .1
#if (!exists("MP_TOP"))    MP_TOP = .9
#if (!exists("MP_GAP"))    MP_GAP = 0.05

#set multiplot layout 2,2 columnsfirst title "{/:Bold=15 Multiplot with explicit page margins}" margins screen MP_LEFT, MP_RIGHT, MP_BOTTOM, MP_TOP spacing screen MP_GAP

set xlabel 'freq'
set log y
#set log x
set autoscale
#set xrange[0:1024]
#set yrange[1e-6:1]
plot "<paste indices.txt diff.txt" w lines lt 1, "<paste indices.txt wdiff.txt" w lines lt 2, "<paste indices.txt filter.txt" w lines lt 3

