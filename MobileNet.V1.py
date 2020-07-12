# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 07:31:28 2020

@author: IMRAN KHAN
"""


model = MobileNet(input_shape = (224,224))
model.add(GlobalAveragePooling2D())
model.add(BatchNormalization(axis = -1, momentum = .99, epsilon = .001))
model.add(Dense(128, activation = "relu"))
model.add(BatchNormalization(axis = -1, momentum = .99, epsilon = .001))
model.add(Dense(16, activation = "relu"))
model.add(BatchNormalization(axis = -1, momentum = .99, epsilon = .001))
model.add(Dense(1))



train_set = ""
test_set = ""
valid_set = ""

model.fit(training_set = train_set, validation_data = valid_set, epochs = 50)
y_pred1 = model.predict(test_set)
y_pred2 = model.predict(valid_set)