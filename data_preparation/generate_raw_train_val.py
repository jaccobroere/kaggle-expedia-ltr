from project_modules.preprocessing import *
import os


def main():
    # wd = r"C:\Users\jacbr\OneDrive\Documenten\vu-data-mining-techniques\Assignment 2"
    wd = r"C:\Users\Beheerder\Documents\vu-data-mining-techniques\Assignment 2"
    os.chdir(wd)

    raw_train = add_target(load_raw_data(train=True))
    train, val = train_val_split_group(raw_train)
    val.drop(
        columns=[
            "position",
            "click_bool",
            "booking_bool",
            "gross_bookings_usd",
        ],
        inplace=True,
    )
    train.to_csv(r"data\train_dev.csv", index=False)
    val.to_csv(r"data\val_dev.csv", index=False)


if __name__ == "__main__":
    main()
