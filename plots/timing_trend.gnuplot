set terminal svg size 350,225 dynamic enhanced font 'Verdana,10'
set output 'plots/timing_trend.svg'

set datafile separator ','

set title "Compilation Time vs Number of Terms"
set xlabel "Number of Terms"
set ylabel "Compilation Time"

plot 'total_time.csv' using 1:2 with points title 'Compilation Time'