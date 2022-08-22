function make_image_grid(arr; n_row=8)
    if length(size(arr)) != 4
        throw(ArgumentError("Expected array with 4 dimensions!"))
    end

    rows = Int64.(ceil(size(arr, 4)/n_row))
    width = size(arr, 1)
    height = size(arr, 2)
    result = zeros(width * n_row, height * rows, size(arr, 3))

    for index in 1:size(arr, 4)
        row = (index - 1) ÷ n_row 
        column = (index - 1) % n_row

        result[1 + column * width:(column + 1) * width, 1 + row * height:(row + 1)*height] = arr[:, :, :, index]
    end
    
    return permutedims(result, (2,1,3))
end