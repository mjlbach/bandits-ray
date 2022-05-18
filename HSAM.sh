for SEED in 1 2
do
    for MAX_DUMMIES in 0 75
    do
        python train.py --model HSAM --min_dummies 0 --max_dummies $MAX_DUMMIES --seed $SEED
    done
done
