import matplotlib.pyplot as plt

# plot activation map
def show_image(activation_map, name=None):
    plt.figure()
    plt.imshow(activation_map)
    plt.axis('off')
    plt.title(name)
    plt.tight_layout()
