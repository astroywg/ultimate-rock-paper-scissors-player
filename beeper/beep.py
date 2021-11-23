import winsound
import time

tune = [0, 0, 4, 0, 0, 7, 0, 0, 12]

i = 0
while i <= 608:
    print(i)
    
    freq = 523.25 * 2 ** (tune[i % len(tune)]/12)

    winsound.Beep(round(freq), 100)
    time.sleep(0.9)
    i += 1


