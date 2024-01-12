export API_KEY=84bec9a1ca5c364023b8e490b7fc3547

for num_samples in 5 10 15 20 25 30
do
  for run in {1..20}
  do
    echo num_samples="$num_samples", run="$run"
    python num_samples_effect.py --num_samples=$num_samples
  done
done