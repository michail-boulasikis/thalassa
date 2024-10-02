set terminal eps font "Arial,8" size 3.3in,2in
set output 'plots/timing_frontend_phases.eps'

set datafile separator ','

set style data histogram
set style histogram rowstacked
set style fill solid border -1
set boxwidth 0.5


set multiplot layout 2,1

# Relative plot
set key box bottom outside samplen 3 spacing 1 width -1 maxcolumns 0
set yrange [0:100]
set ylabel "% of total" offset 2,0
set title "(A) Relative Time Breakdown by Phase" offset 0,-0.5
set xtics nomirror rotate by -45 scale 0
set size 1.2, 0.5
set origin -0.2,0.55

plot 'frontend_measurements.csv' using (100.*$1/$7):xtic(9) t column(1), \
     for [i=2:6] '' using (100.*column(i)/column(7)) title column(i)

# Absolute plot
unset key
unset yrange
set ylabel "Time (s)"
set title "(B) Absolute Time Breakdown by Phase"
set size 1.2, 0.6
set origin -0.2,0
set y2label "Number of Terms"
set y2tics 0, 5, 20
set y2range [0:20]
set key top left inside
plot 'frontend_measurements.csv' skip 1 using 1:xtic(9) notitle, \
            for [i=2:6] '' skip 1 using i:xtic(9) notitle, \
        '' skip 1 using 8 axes x1y2 with points ps 0.5 title "Number of Terms"
unset multiplot