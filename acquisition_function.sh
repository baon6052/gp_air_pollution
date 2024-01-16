export API_KEY=84bec9a1ca5c364023b8e490b7fc3547

for acquisition_func in ModelVariance IVR
do
  for run in {1..5}
  do
    echo acquisition_func="$acquisition_func", run="$run"
    python gaussian_process.py --no-plotting --acquisition_function=$acquisition_func
  done
done