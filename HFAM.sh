for SEED in 1 2 3
do
    for MAX_DUMMIES in 0 20 75
    do
        python train.py --model HFAM --min_dummies 0 --max_dummies $MAX_DUMMIES --seed $SEED
    done
done
