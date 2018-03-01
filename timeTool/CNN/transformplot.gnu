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

set multiplot layout 2,2

set xlabel 'freq'
set log y
set autoscale
set xrange[1:512]
plot 'abs.txt' w lines lt 1
unset log y

set xrange[1:512]
plot 'arg.txt' w lines lt 2

unset ylabel

set autoscale
set xlabel 'pixel'
plot 'sig.txt' w lines lt 3

set autoscale
plot 'back.txt' w lines lt 4
unset multiplot

