
def Myplot(file_list):
    color = ['r','g','b','c','m','y','k','w']
    names = ['Scenario E','Scenario B','Scenario C','Scenario D','Scenario E']
    import matplotlib.pyplot as plt
    i = 0
    for i in range(len(file_list)):
        
        file = open(file_list[i],"r+") 
        lines = file.readlines()
        file.close()
        x = []
        y1 = []
        y2 = []
        y3 = []
        j = 0
        for j in range(len(lines)):
            temp = lines[j].split()
            x.append(float(temp[0]))
            y1.append(float(temp[1]))
            y2.append(float(temp[2]))
            y3.append(float(temp[3]))
        
        plt.plot(x,y1,color=color[i],label =names[i])
        plt.xlabel('n_steps')
        plt.ylabel('Cumulative Rewards')
        plt.title('Training rewards')
        plt.legend()
        
    plt.show()

