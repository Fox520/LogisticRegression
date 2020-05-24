using Pkg
Pkg.add("CSV")
Pkg.add("DataFrames")
Pkg.add("StatsBase")
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
pdays_vector = Array{Float64,1}(undef, size(m)[1]) #new
nr_emp_vector = Array{Float64,1}(undef, size(m)[1]) #new
euribor3m_vector = Array{Float64,1}(undef, size(m)[1]) #new
conf_idx_vector = Array{Float64,1}(undef, size(m)[1]) #new
price_idx_vector = Array{Float64,1}(undef, size(m)[1]) #new

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
        elseif j == 13
            pdays_vector[i] = m[i,j]
        elseif j == 20
            nr_emp_vector[i] = m[i,j]
        elseif j == 19
            euribor3m_vector[i] = m[i,j]
        elseif j == 18
            conf_idx_vector[i] = m[i,j]
        elseif j == 17
            price_idx_vector[i] = m[i,j]
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
        # Works since dataset has "no" as first and not "yes",
        # not the best way but is suitable for now
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
cleaned_pdays = map_to_int(pdays_vector, Float64) #new
cleaned_nr_emp = map_to_int(nr_emp_vector, Float64) #new
cleaned_euribor3m = map_to_int(euribor3m_vector, Float64) #new
cleaned_conf_idx = map_to_int(conf_idx_vector, Float64) #new
cleaned_price_idx = map_to_int(price_idx_vector, Float64) #new


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
            
        elseif j == 13
            m[i,j] = cleaned_pdays[i]
        elseif j == 20
            m[i,j] = cleaned_nr_emp[i]
        elseif j == 19
            m[i,j] = cleaned_euribor3m[i]
        elseif j == 18
            m[i,j] = cleaned_conf_idx[i]
        elseif j == 17
            m[i,j] = cleaned_price_idx[i]
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

# Standardize x ZScoreTransform  UnitRangeTransform
# m_fit = fit(ZScoreTransform, x, dims=2)#, center=false, scale=true)
m_fit = fit(UnitRangeTransform, x, dims=2)
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
    z = (transpose(v_theta) * x)
    # Sigmoid
    return 1/(1+exp(-z))
end

function cost_function(X, Y, theta, lambda)
    m = size(X)[1]
    cross_result = 0 # result of cross-entropy function
    for i in 1:m
        y = Y[i]
        x = X[i, :]
        cross_result += y * log(hypothesis(theta, x))
        cross_result += (1-y)*log(1-hypothesis(theta, x))
    end
    # regularize
    cross_result += (lambda/2*m)*sum(theta[2: size(theta)[1]].^2)
    return -(1/m)*cross_result
end
cost_function(training_x, training_y, zeros(size(training_x)[2]), 0.3)

function update_theta(X, Y, theta, lr, lambda)
    m = size(X)[1]
    for j in 1:size(theta)[1]
        if j == 1
            temp1 = 0 # summation result when theta index 0
            for i in 1:m
                x = X[i, :]
                y = Y[i:i, :][1]
                temp1 += hypothesis(theta, x) - y
                temp1 *= x[1]
            end
            theta[1] = theta[1] - lr*(1/m)*temp1
        else
            temp2 = 0 # summation result when theta index not 0
            for i in 1:m
                x = X[i, :]
                y = Y[i:i, :][1]
                temp2 += hypothesis(theta, x) - y
                temp2 *= x[j]
                temp2 -= (lambda/m) * theta[j]
            end
            theta[j] = theta[j] - lr*(1/m)*temp2 # old - step size
        end
    end
    return theta
end
w=0.002
l=0.3
hey = update_theta(training_x, training_y,zeros(size(training_x)[2]), w, l)
update_theta(training_x, training_y,hey, w, l)

function train(X, Y, theta, lr, lambda, n_iters)
    cost_history = zeros(0)
    for i in 1:n_iters
        cost = cost_function(X, Y, theta, lambda)
        theta = update_theta(X, Y, theta, lr, lambda)
        # Logging purposes
#         append!(cost_history, cost)
        if i % 100 == 0
            println("Cost: $cost Iteration: $i")
        end
    end
    return theta
end
# trained_theta = train(training_x, training_y, zeros(size(training_x)[2]), 0.001, 0.3, 10)

function test(test_x, test_y, theta)
    N = size(test_x)[1]
    accuracy_raw = 0
    zero_count = 0
    one_count = 0
    total_one = 0
    for i in 1:N
        y_pred = hypothesis(theta, test_x[i, :])
        actual = test_y[i]
        if actual == 0 && y_pred < 0.5
            accuracy_raw += 1
            zero_count += 1
        elseif actual == 1 && y_pred >= 0.5
            accuracy_raw += 1
            one_count += 1
        end
        if actual == 1
            total_one += 1
        end
    end
    accuracy = accuracy_raw/N * 100
    precision = one_count/total_one*100
    println()
    println("Accuracy: $accuracy")
    println("Precision: $precision")
    println("Recall: $one_count")
    println()
    println("Actual: 0 => Predicted: $zero_count")
    println("Actual: 1 => Predicted: $one_count")
    
end
println("training started")
trained_theta = train(training_x, training_y, zeros(size(training_x)[2]), 0.002, 0.3, 100)

test(testing_x, testing_y, trained_theta)
