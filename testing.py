# Creating interface to input values and see predicitons
while True:
  number=int(input("Enter a number and see the models prediction compared to the actual image "))-1
  plt.imshow(test_images[number])
  plt.show()
  max_index=np.argmax(predictions[number])
  print("The predicted number is",max_index)
