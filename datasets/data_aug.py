class train_aug():
    def __init__(self) -> None:
        self.color = "RGB"
        self.gamma_low = 0.6
        self.gamma_high = 1.4
        self.brightness_low = 0.5
        self.brightness_high = 1.4
        self.color_low = 0.6
        self.color_high = 1.4
        self.prob = 0.5

    def color_aug(self, left, right):
        if np.random.uniform(0, 1, 1) < self.prob:
            # randomly shift gamma
            random_gamma = np.random.uniform(self.gamma_low, self.gamma_high)
            left_aug = left ** random_gamma
            right_aug = right ** random_gamma

            # randomly shift brightness
            random_brightness = np.random.uniform(
                self.brightness_low, self.brightness_high
            )
            left_aug = left_aug * random_brightness
            right_aug = right_aug * random_brightness

            # randomly shift color
            random_colors = np.random.uniform(self.color_low, self.color_high, 3)
            left_aug *= random_colors
            right_aug *= random_colors

            # saturate
            left = np.clip(left_aug, 0, 255)
            right = np.clip(right_aug, 0, 255)

        return left, right