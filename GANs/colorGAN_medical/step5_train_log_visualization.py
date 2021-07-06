import matplotlib.pyplot as plt

# analyze training logs for models
file=open('logs/hand256_log.txt','r')

ids=[]
d_loss=[]
a_loss=[]

counter=0
for line in file.readlines():
    if line.startswith('discriminator loss:'):
        counter=counter+1
        ids.append(counter)
        d_loss.append(round(float(line.split(':')[1].strip()),4))
    if line.startswith('adversarial loss:'):
        a_loss.append(round(float(line.split(':')[1].strip()),4))

print(ids)
print(d_loss)
print(a_loss)

ids=ids[:100]
d_loss=d_loss[:100]
a_loss=a_loss[:100]

plt.plot(ids,d_loss,label='d_loss')
plt.plot(ids,a_loss,label='a_loss')
# plt.plot(d_loss,a_loss)
plt.legend()
plt.show()


