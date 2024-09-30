set terminal svg size 1200,300
set output 'plots/timing_frontend_phases.svg'

set datafile separator ','

set style data histogram
set style histogram rowstacked
set style fill solid border -1
set boxwidth 0.5



set multiplot layout 1,2 margins 0.1, 0.95, 0.35, 0.95 spacing 0.15, 0.15


myPlotWidth = 0.55

# Relative plot
set key at screen 0.55, 0.85 box samplen 0.4 width -10 spacing 1.0
set yrange [0:100]
set ylabel "% of total"
set title "Relative Time Breakdown"
set xtics nomirror rotate by -45 scale 0
set size ratio myPlotWidth 1

plot 'frontend_measurements.csv' using (100.*$1/$7):xtic(9) t column(1), \
     for [i=2:6] '' using (100.*column(i)/column(7)) title column(i)

# Absolute plot
unset key
unset yrange
set ylabel "Time (s)"
set title "Absolute Time Breakdown"
set size ratio myPlotWidth 1

plot 'frontend_measurements.csv' using 1:xtic(9) t column(1), \
     for [i=2:6] '' using i title column(i)

unset multiplot