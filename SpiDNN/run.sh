rm -r reports/*
python3 main.py

# ugly solution for moving data from horribly named time stamp
# directory to reports for easier use of cat and grep
for dir in reports/*; do
  mv $dir/* reports/
  rmdir $dir
  break
done
