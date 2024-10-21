set terminal postscript eps enhanced color font 'Helvetica,10' size 3in,1.8in
set output 'plots/perf_t.eps'

set style data histogram
set style histogram cluster gap 1
set style fill solid border -1
set boxwidth 0.9

set title "Performance of T-Problems"
set xlabel "Problem"
set ylabel "Time (seconds)"
set xtics rotate by -45

set key outside

# Read data from CSV file
set datafile separator ","
plot 'generated-data/time_measurements.csv' every ::1::6 using 2:xtic(1) title "C Solver", \
        '' every ::1::6 using 3 title "Thalassa CPU", \
     '' every ::1::6 using 4 title "Thalassa GPU"
