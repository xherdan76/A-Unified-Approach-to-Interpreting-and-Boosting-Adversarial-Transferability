import os

import numpy as np


def evaluate(args):

    adv_save_name = f'score_record_{args.target_arch}.npy'
    score_record_path = os.path.join(args.loss_root, adv_save_name)

    try:
        score_record = np.load(score_record_path, allow_pickle=True).item()
    except FileNotFoundError:
        print(
            f'Score file of target {args.target_arch} lam {args.lam} is not found'
        )
        return

    image_names = list(score_record.keys())
    leave_one_out_transferability = np.zeros(len(image_names))
    for i in range(len(image_names)):
        exception_image_name = image_names[i]
        scores = np.array([
            score_record[image_name] for image_name in image_names
            if image_name != exception_image_name
        ])

        image_num = scores.shape[0]
        transferability = (scores > 0).sum(0) / image_num
        best_epoch = np.argmax(transferability).item()
        leave_one_out_transferability[i] = int(
            score_record[exception_image_name][best_epoch] > 0)
    print(
        f'Transferability of source {args.arch} lam {args.lam} on {args.target_arch} is {leave_one_out_transferability.mean()}'
    )
