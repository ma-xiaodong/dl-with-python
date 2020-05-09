import os, shutil
from keras import layers
from keras import models
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

TRAIN_SZ=2000
VAL_SZ=500
TEST_SZ=500
BATCH_SZ=20
EPOCHS=10

def copy_pics(original_cats_dir, original_dogs_dir, 
              train_dir, validation_dir, test_dir):
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)

    train_cats_dir = os.path.join(train_dir, 'cats')
    train_dogs_dir = os.path.join(train_dir, 'dogs')
    validation_cats_dir = os.path.join(validation_dir, 'cats')
    validation_dogs_dir = os.path.join(validation_dir, 'dogs')
    test_cats_dir = os.path.join(test_dir, 'cats')
    test_dogs_dir = os.path.join(test_dir, 'dogs')
    os.mkdir(train_cats_dir)
    os.mkdir(train_dogs_dir)
    os.mkdir(validation_cats_dir)
    os.mkdir(validation_dogs_dir)
    os.mkdir(test_cats_dir)
    os.mkdir(test_dogs_dir)

    fnames = ['{}.jpg'.format(i) for i in range(TRAIN_SZ)]
    # copy train pics
    for fname in fnames:
        # copy cats
        src = os.path.join(original_cats_dir, fname)
        dst = os.path.join(train_cats_dir, fname)
        shutil.copyfile(src, dst)

        # copy dogs
        src = os.path.join(original_dogs_dir, fname)
        dst = os.path.join(train_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # copy validation pics
    fnames = ['{}.jpg'.format(i) for i in range(TRAIN_SZ, TRAIN_SZ+VAL_SZ)]
    for fname in fnames:
        # copy cats
        src = os.path.join(original_cats_dir, fname)
        dst = os.path.join(validation_cats_dir, fname)
        shutil.copyfile(src, dst)

        # copy dogs
        src = os.path.join(original_dogs_dir, fname)
        dst = os.path.join(validation_dogs_dir, fname)
        shutil.copyfile(src, dst)

    # copy test pics
    fnames = ['{}.jpg'.format(i) 
               for i in range(TRAIN_SZ+VAL_SZ, TRAIN_SZ+VAL_SZ+TEST_SZ)]
    for fname in fnames:
        # copy cats
        src = os.path.join(original_cats_dir, fname)
        dst = os.path.join(test_cats_dir, fname)
        shutil.copyfile(src, dst)

        # copy dogs
        src = os.path.join(original_dogs_dir, fname)
        dst = os.path.join(test_dogs_dir, fname)
        shutil.copyfile(src, dst)

def training(train_dir, validation_dir, test_dir):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', 
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(64, (3, 3), activation='relu')) 
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.summary()
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    train_datagen = ImageDataGenerator(rescale=1./255)
    validation_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(150, 150),
            batch_size=BATCH_SZ,
            class_mode='binary')
    validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(150, 150),
            batch_size=BATCH_SZ,
            class_mode='binary')

    history = model.fit_generator(
            train_generator,
            steps_per_epoch=TRAIN_SZ*2//BATCH_SZ,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=VAL_SZ*2//BATCH_SZ)

if __name__ == '__main__':
    original_cats_dir = '/media/mxd/Document/data/kagglecatsanddogs/PetImages/Cat'
    original_dogs_dir = '/media/mxd/Document/data/kagglecatsanddogs/PetImages/Dog'
    base_dir = '/media/mxd/Document/data/kagglecatsanddogs/PetImages_small'

    train_dir = os.path.join(base_dir, 'train')
    validation_dir = os.path.join(base_dir, 'validation')
    test_dir = os.path.join(base_dir, 'test')

    if os.path.exists(train_dir):
        print('{} exists.'.format(train_dir))
    else:
        print('{} does not exists.'.format(train_dir))
        copy_pics(original_cats_dir, original_dogs_dir, 
                  train_dir, validation_dir, test_dir)

    training(train_dir, validation_dir, test_dir)
