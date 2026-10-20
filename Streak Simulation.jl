using Pkg
Pkg.add("Images")
Pkg.add("Random")
Pkg.add("Plots")

using Images, Random, Plots
# Loading image
img = load(raw"C:\Users\chiso\OneDrive\Documents\McMaster Research Coop Summer 2024\Personal Project\Screenshot 2024-08-12 214901.png")
function add_random_streak!(img; 
                            min_length=30, max_length=100, 
                            min_intensity=0.5, max_intensity=1.0, 
                            min_width=1, max_width=5)
    # Randomize streak properties
    intensity = rand(min_intensity:max_intensity)
    width = rand(min_width:max_width)
    angle = rand(0:360) * π / 180  # Convert to radians

    # Convert intensity to Gray, RGB, or RGBA based on image type
    if eltype(img) <: Gray
        color_value = Gray(intensity)
    elseif eltype(img) <: RGB
        color_value = RGB(intensity, intensity, intensity)
    elseif eltype(img) <: RGBA
        color_value = RGBA(intensity, intensity, intensity, 1.0)  # Full opacity
    else
        error("Unsupported image type: $(eltype(img))")
    end

    # Choose a random starting point
    x1 = rand(1:size(img, 1))
    y1 = rand(1:size(img, 2))

    # Calculate the end point based on the angle and length
    length = rand(min_length:max_length)
    x2 = round(Int, x1 + length * cos(angle))
    y2 = round(Int, y1 + length * sin(angle))

    # Ensure end points are within image bounds
    x2 = clamp(x2, 1, size(img, 1))
    y2 = clamp(y2, 1, size(img, 2))

    # Add the streak to the image
    for i in 0:width-1
        if x1 == x2
            img[x1:x2, y1+i:y2+i] .= color_value
        elseif y1 == y2
            img[x1+i:x2+i, y1:y2] .= color_value
        else
            for t in range(0, stop=1, length=100)
                x = round(Int, x1 * (1-t) + x2 * t)
                y = round(Int, y1 * (1-t) + y2 * t)
                if x+i <= size(img, 1) && y+i <= size(img, 2)
                    img[x+i, y+i] = color_value
                end
            end
        end
    end
    return img
end

# Apply a random streak
img_with_streak = add_random_streak!(img, 
                                     min_length=50, max_length=150, 
                                     min_intensity=0.6, max_intensity=0.9, 
                                     min_width=2, max_width=5)

# Display the image with the random streak
display(plot(img_with_streak, title="Image with Random Streak"))