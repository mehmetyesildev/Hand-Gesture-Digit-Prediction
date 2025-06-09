import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# Veri setini yüklemek için ImageDataGenerator kullanıyoruz
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Görüntülerin pixel değerlerini [0, 1] aralığına çeker
    shear_range=0.2,  # Kesme dönüşümü
    zoom_range=0.2,   # Zoom dönüşümü
    horizontal_flip=True  # Yatay çevirme
)

# Eğitim verilerini yükleyin
train_generator = train_datagen.flow_from_directory(
    'images/',  # Veri setinizin bulunduğu klasör
    target_size=(64, 64),  # Resimleri yeniden boyutlandır
    batch_size=32,
    class_mode='categorical'  # Sınıf etiketleri çoklu sınıf (sayılar)
)

# CNN Modeli Oluşturma
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(len(os.listdir('images/')), activation='softmax')  # Sınıf sayısına göre çıkış katmanı
])

# Modeli derleme
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Modeli eğitme
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=10,
)

# Modeli kaydetme
model.save('model/hand_gesture_model.h5')
