using CSV
using DataFrames
using StatsBase  # For standardization

# Load the dataset
file_path = "bank-additional-full.csv"
df = CSV.read(file_path)

# Convert dataframe to matrix
m = convert(Matrix, df)
# Categorical feature vectors which need mapping to numbers
marital_vector = Array{String,1}(undef, size(m)[1])
default_vector = Array{String,1}(undef, size(m)[1])
loan_vector = Array{String,1}(undef, size(m)[1])
month_vector = Array{String,1}(undef, size(m)[1])
duration_vector = Array{Int64,1}(undef, size(m)[1])
poutcome_vector = Array{String,1}(undef, size(m)[1])
job_vector = Array{String,1}(undef, size(m)[1])
education_vector = Array{String,1}(undef, size(m)[1])
housing_vector = Array{String,1}(undef, size(m)[1])
contact_vector = Array{String,1}(undef, size(m)[1])
dow_vector = Array{String,1}(undef, size(m)[1])
y_vector = Array{String,1}(undef, size(m)[1])

# Map categorical features to numbers
for i in 1:size(m)[1]
    for j in 1:size(m)[2]
        # Check if columns match
        if j == 3
            marital_vector[i] = m[i,j]
        elseif j == 5
            default_vector[i] = m[i,j]
        elseif j == 7
            loan_vector[i] = m[i,j]
        elseif j == 9
            month_vector[i] = m[i,j]
        elseif j == 11
            duration_vector[i] = m[i,j]
        elseif j == 15
            poutcome_vector[i] = m[i,j]
        elseif j == 2
            job_vector[i] = m[i,j]
        elseif j == 4
            education_vector[i] = m[i,j]
        elseif j == 6
            housing_vector[i] = m[i,j]
        elseif j == 8
            contact_vector[i] = m[i,j]
        elseif j == 10
            dow_vector[i] = m[i,j]
        elseif j == 21
            y_vector[i] = m[i,j]
        end
    end
end

function map_to_int(arr, T, initial_count=nothing)
    # Dynamically map the strings to numbers
    unique_list = T[]
    for v in arr
        if v in unique_list
            continue
        else
            push!(unique_list, v)
        end
    end
    cleaned_list = Array{Int64, 1}(undef, length(arr))
    if initial_count == nothing
        count = 1
    else
        count = initial_count
    end
    for i in unique_list
        for j in 1:length(arr)
            if arr[j] == i
                cleaned_list[j] = count
            end
        end
        count += 1
    end
    return cleaned_list

end

cleaned_marital = map_to_int(marital_vector, String)
cleaned_default = map_to_int(default_vector, String)
cleaned_loan = map_to_int(loan_vector, String)
cleaned_month = map_to_int(month_vector, String)

cleaned_duration = map_to_int(duration_vector, Int64)
cleaned_poutcome = map_to_int(poutcome_vector, String)
cleaned_job = map_to_int(job_vector, String)
cleaned_education = map_to_int(education_vector, String)

cleaned_housing = map_to_int(housing_vector, String)
cleaned_contact = map_to_int(contact_vector, String)
cleaned_dow = map_to_int(dow_vector, String)
# Special case where we want only 1 and 0 
y = map_to_int(y_vector, String, 0)

# Update the matrix with the numerical mapped equivalents
for i in 1:size(m)[1]
    for j in 1:size(m)[2]
        if j == 3
            m[i,j] = cleaned_marital[i]
        elseif j == 5
            m[i,j] = cleaned_default[i]
        elseif j == 7
            m[i,j] = cleaned_loan[i]
        elseif j == 9
            m[i,j] = cleaned_month[i]
        elseif j == 11
            m[i,j] = cleaned_duration[i]
        elseif j == 15
            m[i,j] = cleaned_poutcome[i]
        elseif j == 2
            m[i,j] = cleaned_job[i]
        elseif j == 4
            m[i,j] = cleaned_education[i]
        elseif j == 6
            m[i,j] = cleaned_housing[i]
        elseif j == 8
            m[i,j] = cleaned_contact[i]
        elseif j == 10
            m[i,j] = cleaned_dow[i]
        end
    end
end

# Remove y from feature matrix
x = Matrix{Float64}(undef, size(m)[1], size(m)[2]-1)

for i in 1:size(m)[1]
    for j in 1:size(m)[2]-1
        # Merge the outcome value as the last index
        if j == size(m)[2]-1
            x[i,j] = y[i]
            continue
        end
        x[i,j] = m[i,j]
    end
end

# Concatenate a vector of 1 to feature matrix to represent bias
x = hcat(x, ones(Int64, size(x)[1]))

# Standardize x
m_fit = fit(ZScoreTransform, x, dims=2)
x = StatsBase.transform(m_fit, x)

# Split into testing and training
TRAIN_PERCENT = 0.8
# Calculate number of rows to use as train sample
training_row_length = trunc(Int64, size(x)[1] * TRAIN_PERCENT)
training_x = x[1:training_row_length-1, :]
testing_x = x[training_row_length: size(x)[1], :]

training_y = y[1:training_row_length-1, :]
testing_y = y[training_row_length: size(y)[1], :]

function hypothesis(v_theta, x)
    z = transpose(v_theta) * x
    # Sigmoid
    return 1/(1+exp(-z))
end


