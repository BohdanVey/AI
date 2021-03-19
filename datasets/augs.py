import albumentations as albu


class StandardAugmentation:
    def __init__(self, p=0.6):
        self.p = p
        self.aug = self.__build_augmentator()

    def __call__(self, image):
        res = self.aug(image=image)
        return res["image"]

    def __build_augmentator(self):
        return albu.Compose([
            albu.OneOf([
                albu.VerticalFlip(p=0.5),
                albu.HorizontalFlip(p=0.5),
            ], p=0.5),
        ], p=self.p)


class SmartCrop:
    def __init__(self, height, width, p=1.0):
        self.p = p
        self.height = height
        self.width = width
        self.aug = self.__build_augmentator()

    def __call__(self, image, mask):
        res = self.aug(image=image, mask=mask)
        return res['image'], res['mask']

    def __build_augmentator(self):
        return albu.Compose([
            albu.CropNonEmptyMaskIfExists(height=self.height, width=self.width, p=1.0),
            albu.OneOf([
                albu.VerticalFlip(p=0.5),
                albu.HorizontalFlip(p=0.5),
            ], p=0.5),
        ], p=self.p)


class CenterCrop:
    def __init__(self, height, width, p=1.0):
        self.p = p
        self.height = height
        self.width = width
        self.aug = self.__build_augmentator()

    def __call__(self, image, mask):
        res = self.aug(image=image, mask=mask)
        return res['image'], res['mask']

    def __build_augmentator(self):
        return albu.Compose([
            albu.CenterCrop(256, 256, p=1.0)
        ], p=1.0)


class SmartCropColorAndScale:
    def __init__(self, height, width, p=1.0):
        self.p = p
        self.height = height
        self.width = width
        self.aug = self.__build_augmentator()

    def __call__(self, image, mask):
        res = self.aug(image=image, mask=mask)
        return res['image'], res['mask']

    def __build_augmentator(self):
        return albu.Compose([
            albu.CropNonEmptyMaskIfExists(height=self.height, width=self.width, p=1.0),
            albu.ShiftScaleRotate(shift_limit=0.15, scale_limit=0.15, p=0.6),
            albu.PadIfNeeded(256, 256),
            albu.OneOf([
                albu.VerticalFlip(p=0.5),
                albu.HorizontalFlip(p=0.5),
            ], p=0.5),
            albu.RandomBrightnessContrast(0.1, 0.1),
            # albu.RandomGamma()
        ], p=self.p)