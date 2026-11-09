# Titanic survival prediction with logistic regression (R / glm).
# Reads train.csv and test.csv from DATA_DIR (default: ../data) and prints every step.

suppressPackageStartupMessages(library(dplyr))

data_dir <- Sys.getenv("DATA_DIR", unset = NA)
if (is.na(data_dir)) {
  file_arg <- sub("--file=", "", grep("--file=", commandArgs(trailingOnly = FALSE), value = TRUE))
  script_dir <- if (length(file_arg) > 0) dirname(normalizePath(file_arg)) else getwd()
  data_dir <- file.path(script_dir, "..", "data")
}

banner <- function(text) cat("\n", strrep("=", 70), "\n", text, "\n", strrep("=", 70), "\n", sep = "")
accuracy <- function(actual, predicted) mean(actual == predicted)

title_levels <- c("Mr", "Mrs", "Miss", "Master", "Rare")

get_title <- function(name) {
  title <- trimws(sub("^.*,\\s*([^.]+)\\..*$", "\\1", name))
  title[title %in% c("Mlle", "Ms")] <- "Miss"
  title[title == "Mme"] <- "Mrs"
  title[!title %in% title_levels[1:4]] <- "Rare"
  title
}

# Create new columns and fill missing values using stats from the training set
add_features <- function(df, ref) {
  df <- df %>%
    mutate(
      Title = factor(get_title(Name), levels = title_levels),
      FamilySize = SibSp + Parch + 1,
      IsAlone = as.integer(FamilySize == 1)
    )
  age_fill <- unname(ref$age_by_title[as.character(df$Title)])
  age_fill[is.na(age_fill)] <- ref$age_all
  df$Age <- ifelse(is.na(df$Age), age_fill, df$Age)
  df$Fare[is.na(df$Fare)] <- ref$fare
  df$Embarked[is.na(df$Embarked)] <- ref$embarked
  df %>% mutate(
    Pclass = factor(Pclass, levels = 1:3),
    Sex = factor(Sex, levels = c("female", "male")),
    Embarked = factor(Embarked, levels = c("C", "Q", "S"))
  )
}

banner("STEP 1: Load data")
cat("Looking for data in:", data_dir, "\n")
train_path <- file.path(data_dir, "train.csv")
test_path <- file.path(data_dir, "test.csv")
if (!file.exists(train_path) || !file.exists(test_path)) {
  stop("train.csv and/or test.csv not found. Download them from Kaggle into the data folder (see README).")
}
train <- read.csv(train_path, na.strings = c("", "NA"), stringsAsFactors = FALSE)
test <- read.csv(test_path, na.strings = c("", "NA"), stringsAsFactors = FALSE)
cat("train.csv:", nrow(train), "rows,", ncol(train), "columns\n")
cat("test.csv: ", nrow(test), "rows,", ncol(test), "columns\n")
cat("\nFirst 5 rows of train.csv:\n")
print(head(train, 5))

banner("STEP 2: Explore the training data")
cat(sprintf("Overall survival rate: %.1f%%\n", 100 * mean(train$Survived)))
cat("\nSurvival rate by sex:\n")
print(round(tapply(train$Survived, train$Sex, mean), 3))
cat("\nSurvival rate by passenger class:\n")
print(round(tapply(train$Survived, train$Pclass, mean), 3))
cat("\nMissing values in train.csv (columns with any):\n")
missing <- colSums(is.na(train))
print(missing[missing > 0])

banner("STEP 3: Clean and engineer features")
ref <- list(
  age_by_title = tapply(train$Age, get_title(train$Name), median, na.rm = TRUE),
  age_all = median(train$Age, na.rm = TRUE),
  fare = median(train$Fare, na.rm = TRUE),
  embarked = names(which.max(table(train$Embarked)))
)
cat("Added Title (extracted from Name; rare titles grouped as 'Rare')\n")
cat("Added FamilySize = SibSp + Parch + 1 and IsAlone\n")
cat(sprintf("Filled missing Age with the median age of the passenger's Title (overall median %.1f)\n", ref$age_all))
cat(sprintf("Filled missing Fare with the train median (%.2f)\n", ref$fare))
cat("Filled missing Embarked with the most common port (", ref$embarked, ")\n", sep = "")
train <- add_features(train, ref)
cat("Remaining missing values in model columns:", sum(is.na(train[, c("Age", "Fare", "Embarked")])), "\n")
cat("Dropped Cabin, Ticket, Name and PassengerId (not used by the model)\n")
cat("Converted Pclass, Sex, Embarked and Title to factors (glm one-hot encodes them)\n")

banner("STEP 4: Fit logistic regression")
# Sex is left out on purpose: Title (Mr/Mrs/Miss/Master) already encodes it and the overlap
# makes glm's coefficients blow up (near-perfect collinearity).
model_formula <- Survived ~ Pclass + Age + Fare + Embarked + Title + FamilySize + IsAlone
set.seed(42)
hold_idx <- sample(nrow(train), size = round(0.2 * nrow(train)))
fit_df <- train[-hold_idx, ]
hold_df <- train[hold_idx, ]
cat(sprintf("Split train.csv 80/20 -> %d fit rows, %d hold-out rows (seed 42)\n", nrow(fit_df), nrow(hold_df)))
model <- glm(model_formula, data = fit_df, family = binomial)
cat("\nLargest coefficients:\n")
coefs <- coef(model)
coefs <- coefs[names(coefs) != "(Intercept)"]
print(round(head(coefs[order(-abs(coefs))], 6), 3))

banner("STEP 5: Accuracy")
predict_class <- function(m, df) as.integer(predict(m, newdata = df, type = "response") > 0.5)
cat(sprintf("Training accuracy:          %.3f\n", accuracy(fit_df$Survived, predict_class(model, fit_df))))
cat(sprintf("Hold-out accuracy:          %.3f  (20%% of train.csv, never used to fit)\n",
            accuracy(hold_df$Survived, predict_class(model, hold_df))))
# Refit on all labeled data for the final test predictions
model <- glm(model_formula, data = train, family = binomial)
cat(sprintf("Training accuracy (refit on all %d rows): %.3f\n", nrow(train),
            accuracy(train$Survived, predict_class(model, train))))

banner("STEP 6: Predict on test.csv")
test_feat <- add_features(test, ref)
preds <- predict_class(model, test_feat)
cat(sprintf("Predicted %d passengers: %d survive, %d do not\n", length(preds), sum(preds), sum(1 - preds)))
out_path <- file.path(data_dir, "predictions_r.csv")
write.csv(data.frame(PassengerId = test$PassengerId, Survived = preds), out_path, row.names = FALSE)
cat("Saved predictions to", out_path, "\n")

if ("Survived" %in% names(test)) {
  cat(sprintf("Test accuracy: %.3f\n", accuracy(test$Survived, preds)))
} else {
  cat("Note: Kaggle's test.csv has no Survived column, so test accuracy cannot be computed here.\n")
  cat("The hold-out accuracy above is the honest estimate of performance on unseen passengers.\n")
  cat("Upload predictions_r.csv to Kaggle for the official test score.\n")
}
