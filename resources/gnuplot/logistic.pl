set terminal svg font "Bitstream Vera Sans,24" size 620,420
set output "Logistic-curve.svg"

set xrange [-6:6]
set xzeroaxis linetype -1
set yzeroaxis linetype -1
set xtics axis nomirror
set ytics axis nomirror 0,0.5,1
set key off
set grid lw 0.5 lt 1 lc black
set border linewidth 2
set border 1

set samples 400

plot exp(x)/(1 + exp(x)) with line linetype rgbcolor "medium-blue" linewidth 2
