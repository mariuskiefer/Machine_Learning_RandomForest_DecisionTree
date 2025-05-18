library(shiny)
library(shinythemes)
library(caret)
library(rpart)
library(rpart.plot)
library(randomForest)
library(ggplot2)
library(dplyr)
library(pROC)
library(plotly)

# Load dataset
cancer <- read.csv("trimmed_dataset.csv")
colnames(cancer) <- trimws(colnames(cancer))
# Ensure diagnosis is treated as a factor for classification
cancer$diagnosis <- as.factor(cancer$diagnosis)

# Set seed for reproducibility
set.seed(123)

# Split data
trainIndex <- createDataPartition(cancer$diagnosis, p = 0.7, list = FALSE)
train_data <- cancer[trainIndex, ]
test_data <- cancer[-trainIndex, ]

# Get predictor columns
predictor_cols <- setdiff(names(cancer), "diagnosis")
predictor_groups <- split(predictor_cols, ceiling(seq_along(predictor_cols) / 5))

# Define color palette for consistency
colors <- list(
  primary_blue = "#3498db",
  dark_blue = "#2c3e50",
  malignant = "#e74c3c",
  benign = "#2ecc71",
  background = "#f8f9fa",
  border = "#ddd"
)

# UI
ui <- fluidPage(
  # change the theme
  theme = shinytheme("flatly"),
  tags$head(
    # define styles for tabs and pills
    tags$style(HTML("
      .nav-tabs > li > a,
      .nav-pills > li > a {
        color: #2c3e50 !important;
        font-weight: 500;
        white-space: nowrap;
      }
      .nav-tabs > li.active > a,
      .nav-tabs > li.active > a:focus,
      .nav-tabs > li.active > a:hover,
      .nav-pills > li.active > a,
      .nav-pills > li.active > a:focus,
      .nav-pills > li.active > a:hover {
        color: #2c3e50 !important;
        font-weight: bold;
        border-top: 2px solid #2c3e50;
        border-bottom: 2px solid #2c3e50;
        background-color: #f8f9fa !important;  /* keep the background unchanged */
      }
      .toggle-btn {
        font-size: 12px;
        text-decoration: none;
        color: #3498db;
        cursor: pointer;
        background: none;
        border: none;
        padding: 0;
        margin-left: 10px;
      }
      .explanation-box {
        background-color: #f0f3f6;
        padding: 10px;
        border-radius: 5px;
        margin-top: 5px;
        font-size: 13px;
      }
    "))
  ),
  titlePanel("Breast Cancer Prediction with Decision Tree & Random Forest"),
  navbarPage(
    title = "Breast Cancer Predictor",
    # Decision Tree Tab
    tabPanel("Decision Tree",
          # sidebar to change the parameters of decision tree
          sidebarLayout(
            sidebarPanel(
              div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                h4("Hyperparameter Tuning", align = "center"),
                selectInput("splitCriteria", "Select Splitting Criteria:",
                          choices = c("Gini" = "gini", "Entropy" = "information"),
                          selected = "information"),
                sliderInput("cp", "Complexity Parameter:", min = 0.001, max = 0.1, value = 0.01, step = 0.001),
                sliderInput("maxdepth", "Maximum Depth:", min = 1, max = 30, value = 10, step = 1),
                sliderInput("minsplit", "Minimum Split:", min = 2, max = 50, value = 5, step = 1)
              ),
              div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 20px;",
                h4("Understanding Settings", align = "center"),
                tags$ul(
                  tags$li(strong("Split Criteria:"), "Gini or Entropy - Both measure how well patients are grouped. Like two ways of sorting patients, with similar results."),
                  tags$li(strong("Complexity:"), "Controls tree size. Lower values create bigger trees with more detail. Like deciding how specialized your diagnosis should be."),
                  tags$li(strong("Maximum Depth:"), "Limits how many questions are asked. Shallow trees (3-5) focus on key factors, deeper trees consider more detail."),
                  tags$li(strong("Minimum Split:"), "Sets how many patients needed before asking another question. Higher values prevent decisions based on rare cases.")
                ),
                p("Medical example: A simple tree checks basics like cell size and shape. A complex tree might also consider rare patterns only seen in specific patients.")
              )
            ),
            mainPanel(
              tabsetPanel(
                # Explore Model Tab for Decision Tree
                tabPanel("Explore Model",
                  fluidRow(
                    column(12,
                      h3("How a Decision Tree Works", align = "center"),
                      p("This tool uses a decision tree algorithm to help predict whether a breast tumor is benign or malignant. Here's how it works:", style = "font-size: 16px;"),
                      div(style = "display: flex; flex-direction: row; flex-wrap: nowrap; justify-content: center; gap: 10px; overflow-x: auto; margin-bottom: 30px;",
                          div(style = "width: 180px; margin: 0; text-align: center; border: 1px solid #ddd; border-radius: 5px; padding: 10px;",
                              icon("home", "fa-3x", style = "color: #3498db;"),
                              h4("Step 1: Root Node"),
                              p("All data begins at the root node representing the full dataset.")
                          ),
                          div(style = "width: 180px; margin: 0; text-align: center; border: 1px solid #ddd; border-radius: 5px; padding: 10px;",
                              icon("scissors", "fa-3x", style = "color: #3498db;"),
                              h4("Step 2: Splitting"),
                              p("The algorithm selects the best feature to split the data and reduce impurity.")
                          ),
                          div(style = "width: 180px; margin: 0; text-align: center; border: 1px solid #ddd; border-radius: 5px; padding: 10px;",
                              icon("sitemap", "fa-3x", style = "color: #3498db;"),
                              h4("Step 3: Branching"),
                              p("Each split creates branches that recursively partition the data.")
                          ),
                          div(style = "width: 180px; margin: 0; text-align: center; border: 1px solid #ddd; border-radius: 5px; padding: 10px;",
                              icon("flag", "fa-3x", style = "color: #3498db;"),
                              h4("Step 4: Terminal Node"),
                              p("Leaf nodes represent final predictions based on the decision rules.")
                          )
                      )
                    )
                  ),
                  fluidRow(
                    column(12, 
                      div(style = "background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px;",
                          h3("Medical Analogy", align = "center"),
                          p("Think of a Decision Tree like a diagnostic flowchart:", style = "font-size: 16px;"),
                          tags$ul(
                            tags$li("Each branch is like a question guiding the diagnosis."),
                            tags$li("The flowchart directs you from symptoms to a final diagnosis."),
                            tags$li("At every decision point, a specific criterion is applied."),
                            tags$li("The final leaf node delivers the prediction based on accumulated evidence.")
                          )
                      )
                    )
                  ),
                  fluidRow(
                    column(6,
                      div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; height: 100%;",
                        h3("Why Decision Trees Are Effective", align = "center"),
                        tags$ul(
                          tags$li(strong("Simplicity:"), "Easy to understand and interpret."),
                          tags$li(strong("Transparency:"), "Clearly shows the decision-making process."),
                          tags$li(strong("No Data Scaling:"), "Works well with raw data without preprocessing."),
                          tags$li(strong("Fast Predictions:"), "Quickly generates predictions.")
                        )
                      )
                    ),
                    column(6,
                      div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; height: 100%;",
                        h3("How to Use This Tool", align = "center"),
                        tags$ol(
                          tags$li("Adjust hyperparameters in the ", tags$b("Hyperparameter Tuning"), " controls on the left."),
                          tags$li("Review the tree structure in the ", tags$b("Tree Plot"), " tab."),
                          tags$li("Examine performance metrics in the ", tags$b("Performance"), " tab."),
                          tags$li("Check the impact of features in the ", tags$b("Variable Importance"), " tab.")
                        )
                      )
                    )
                  )
                ),
                # visualization of the tree
                tabPanel("Tree Plot", 
                  div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                    h4("Decision Tree Visualization", align = "center"),
                    p("This tree shows how decisions are made. Each node splits patients based on a measurement threshold, leading to final predictions (B: Benign, M: Malignant).")
                  ),
                  plotOutput("treePlot")
                ),
                # performance tab including roc, confusion, and accuracy etc.
                tabPanel("Performance",
                  verticalLayout(
                    div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                      h4("Performance Metrics Explained", align = "center"),
                      tags$ul(
                        tags$li(strong("ROC Curve:"), "Shows trade-offs between catching all malignant tumors vs. wrongly flagging benign ones."),
                        tags$li(strong("AUC:"), "Measures overall accuracy from 0 to 1. Higher is better. Above 0.9 is excellent."),
                        tags$li(strong("Threshold:"), "Adjusts sensitivity. Lower catches more malignant tumors but may increase false alarms.")
                      ),
                      p("Adjust the threshold slider to see how it affects diagnostic accuracy.")
                    ),
                    div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                      h4("ROC Curve (from Binary Predictions)", align = "center"),
                      plotlyOutput("rocPlot"),
                      sliderInput("threshold", "Threshold", min = 0, max = 1, value = 0.5, step = 0.01)
                    ),
                    div(style = "display: flex; flex-wrap: wrap; justify-content: space-between;",
                      div(style = "width: 48%; min-width: 250px; background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                        div(style = "display: flex; align-items: center;",
                          h4("Confusion Matrix", align = "center", style = "margin-right: auto;"),
                          actionButton("dt_conf_toggle", 
                                      ifelse(TRUE, "▼ Show explanation", "▲ Hide explanation"), 
                                      class = "toggle-btn")
                        ),
                        verbatimTextOutput("conf_matrix"),
                        conditionalPanel(
                          condition = "input.dt_conf_toggle % 2 == 1",
                          div(class = "explanation-box",
                            tags$ul(
                              tags$li("Rows: What the model predicted (0=Benign, 1=Malignant)"),
                              tags$li("Columns: Actual diagnosis (0=Benign, 1=Malignant)"),
                              tags$li("Diagonal (top-left to bottom-right): Correct predictions"),
                              tags$li("Off-diagonal: Errors (missed diagnoses)")
                            )
                          )
                        )
                      ),
                      div(style = "width: 48%; min-width: 250px; background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                        div(style = "display: flex; align-items: center;",
                          h4("Performance Metrics", align = "center", style = "margin-right: auto;"),
                          actionButton("dt_metrics_toggle", 
                                      ifelse(TRUE, "▼ Show explanation", "▲ Hide explanation"), 
                                      class = "toggle-btn")
                        ),
                        verbatimTextOutput("performance_metrics"),
                        conditionalPanel(
                          condition = "input.dt_metrics_toggle % 2 == 1",
                          div(class = "explanation-box",
                            tags$ul(
                              tags$li(strong("Accuracy:"), "Percentage of all correct predictions"),
                              tags$li(strong("Precision:"), "When model predicts malignant, how often it's right"),
                              tags$li(strong("Recall:"), "Percentage of actual malignant tumors found"),
                              tags$li(strong("Specificity:"), "Percentage of actual benign tumors correctly identified"),
                              tags$li(strong("F1 Score:"), "Balance between precision and recall (1 is best)")
                            )
                          )
                        )
                      )
                    )
                  )
                ),
                # Variable importance tab to show which variable effect the model
                tabPanel("Variable Importance", 
                  div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                    h4("Feature Importance Explained", align = "center"),
                    p("This chart shows which measurements are most useful for diagnosis. Longer bars mean that feature is more important for predicting malignancy.")
                  ),
                  plotOutput("varImpPlot")
                ),
                # prediction tool with dynamically assigned feature names
                tabPanel("Prediction Tool",
                  sidebarLayout(
                    sidebarPanel(
                      div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                        h4("Enter Patient Data", align = "center"),
                        do.call(tabsetPanel, c(
                          list(id = "featureTabs"),
                          lapply(seq_along(predictor_groups), function(i) {
                            group <- predictor_groups[[i]]
                            tabPanel(
                              title = paste("Features", ((i - 1) * 5 + 1), "to", i * 5),
                              do.call(tagList, lapply(group, function(col) {
                                sliderInput(inputId = col, label = col,
                                            min = min(cancer[[col]], na.rm = TRUE),
                                            max = max(cancer[[col]], na.rm = TRUE),
                                            value = median(cancer[[col]], na.rm = TRUE),
                                            step = (max(cancer[[col]], na.rm = TRUE) - min(cancer[[col]], na.rm = TRUE)) / 100)
                              }))
                            )
                          })
                        )),
                        actionButton("predict_btn", "Predict", class = "btn-primary", style = "width: 100%; margin-top: 10px;")
                      )
                    ),
                    mainPanel(
                      div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                        h4("Prediction Result", align = "center"), 
                        verbatimTextOutput("prediction_result"),
                        plotlyOutput("prediction_prob_plot")
                      )
                    )
                  )
                )
              )
            )
          )
        ),
        
        # Random Forest Tab
        tabPanel("Random Forest",
          sidebarLayout(
            sidebarPanel(
              div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                h4("Hyperparameter Tuning", align = "center"),
                sliderInput("rf_ntree", "Number of Trees:", 
                            min = 10, max = 1000, value = 500, step = 10),
                sliderInput("rf_mtry", "Number of Variables at Each Split:", 
                            min = 1, max = length(predictor_cols), 
                            value = floor(sqrt(length(predictor_cols))), step = 1),
                sliderInput("rf_nodesize", "Minimum Size of Terminal Nodes:", 
                            min = 1, max = 10, value = 5, step = 1),
                checkboxInput("rf_replace", "Bootstrap Sample:", value = TRUE)
              ),
              div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-top: 20px;",
                h4("Understanding Settings", align = "center"),
                tags$ul(
                  tags$li(strong("Number of Trees:"), "Like getting opinions from multiple doctors. More trees (200-500) usually means better results."),
                  tags$li(strong("Variables Per Split:"), "How many factors each 'doctor' considers at once. Default setting works well for most cases."),
                  tags$li(strong("Minimum Node Size:"), "Prevents decisions based on too few patients. Higher values (5-10) create more reliable rules."),
                  tags$li(strong("Bootstrap Sample:"), "When checked, each tree sees slightly different patient data, making the overall prediction more reliable.")
                ),
                p("Medical example: It's like having a team of specialists review a case, each looking at different aspects, then voting on the final diagnosis.")
              )
            ),
            mainPanel(
              tabsetPanel(
                # Explore tab
                tabPanel("Explore Model",
                  fluidRow(
                    column(12,
                      h3("How Random Forest Works", align = "center"),
                      p("This tool uses a machine learning technique called 'Random Forest' to help predict whether a breast tumor is benign or malignant. Here's how it works:", style = "font-size: 16px;"),
                      div(style = "display: flex; flex-direction: row; flex-wrap: nowrap; justify-content: center; gap: 10px; overflow-x: auto; margin-bottom: 30px;",
                          div(style = "width: 180px; margin: 0; text-align: center; border: 1px solid #ddd; border-radius: 5px; padding: 10px;",
                              icon("tree", "fa-3x", style = "color: #3498db;"),
                              h4("Step 1: Many Decision Trees"),
                              p("Random Forest builds many decision trees, each slightly different from the others.")
                          ),
                          div(style = "width: 180px; margin: 0; text-align: center; border: 1px solid #ddd; border-radius: 5px; padding: 10px;",
                              icon("random", "fa-3x", style = "color: #3498db;"),
                              h4("Step 2: Random Sampling"),
                              p("Each tree is trained on a random subset of patient data and features.")
                          ),
                          div(style = "width: 180px; margin: 0; text-align: center; border: 1px solid #ddd; border-radius: 5px; padding: 10px;",
                              icon("check-square", "fa-3x", style = "color: #3498db;"),
                              h4("Step 3: Individual Predictions"),
                              p("Each tree independently predicts whether a tumor is benign or malignant.")
                          ),
                          div(style = "width: 180px; margin: 0; text-align: center; border: 1px solid #ddd; border-radius: 5px; padding: 10px;",
                              icon("users", "fa-3x", style = "color: #3498db;"),
                              h4("Step 4: Majority Vote"),
                              p("The final prediction combines all trees' votes (like a medical team consensus).")
                          )
                      )
                    )
                  ),
                  fluidRow(
                    column(12, 
                      div(style = "background-color: #f8f9fa; padding: 20px; border-radius: 10px; margin-bottom: 30px;",
                          h3("Medical Analogy", align = "center"),
                          p("Think of Random Forest like a tumor board conference:", style = "font-size: 16px;"),
                          tags$ul(
                            tags$li("Each 'tree' is like an individual doctor reviewing patient cases"),
                            tags$li("Each doctor looks at slightly different patient histories and test results"),
                            tags$li("Each doctor independently forms an opinion about malignancy"),
                            tags$li("The final diagnosis comes from combining all doctors' opinions"),
                            tags$li("This approach is more reliable than depending on just one doctor's assessment")
                          )
                      )
                    )
                  ),
                  fluidRow(
                    column(6,
                      div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; height: 100%;",
                        h3("Why Random Forest Is Effective", align = "center"),
                        tags$ul(
                          tags$li(strong("High Accuracy:"), "By combining many decision trees, Random Forest achieves better predictions than a single decision tree."),
                          tags$li(strong("Handles Complex Relationships:"), "Can detect patterns between cell measurements that might not be obvious to the human eye."),
                          tags$li(strong("Manages Missing Data:"), "Works well even with incomplete information."),
                          tags$li(strong("Identifies Key Features:"), "Ranks which cell measurements are most important for diagnosis.")
                        )
                      )
                    ),
                    column(6,
                      div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; height: 100%;",
                        h3("How to Use This Tool", align = "center"),
                        tags$ol(
                          tags$li("Experiment with different model settings in the ", tags$b("Hyperparameter Tuning"), " inputs on the left."),
                          tags$li("Understand the model's overall performance in the ", tags$b("Performance"), " tab."),
                          tags$li("Explore which cellular features are most important for diagnosis in the ", tags$b("Variable Importance"), " tab."),
                          tags$li("Enter cell measurements from a patient's sample in the ", tags$b("Prediction Tool"), " tab to get a malignancy prediction.")
                        )
                      )
                    )
                  )
                ),
                # performance tab including roc, confusion, accuracy etc. 
                tabPanel("Performance", 
                  verticalLayout(
                    div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                      h4("Performance Metrics Explained", align = "center"),
                      tags$ul(
                        tags$li(strong("ROC Curve:"), "Shows trade-offs between catching all malignant tumors vs. wrongly flagging benign ones."),
                        tags$li(strong("AUC:"), "Measures overall accuracy from 0 to 1. Higher is better. Above 0.9 is excellent."),
                        tags$li(strong("Threshold:"), "Adjusts sensitivity. Lower catches more malignant tumors but may increase false alarms.")
                      ),
                      p("Random Forest typically achieves better accuracy than a single Decision Tree.")
                    ),
                    div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                      h4("RF ROC Curve (from Binary Predictions)", align = "center"),
                      plotlyOutput("rf_rocPlot"),
                      sliderInput("rf_threshold", "Threshold", min = 0, max = 1, value = 0.5, step = 0.01)
                    ),
                    div(style = "display: flex; flex-wrap: wrap; justify-content: space-between;",
                      div(style = "width: 48%; min-width: 250px; background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                        div(style = "display: flex; align-items: center;",
                          h4("Confusion Matrix", align = "center", style = "margin-right: auto;"),
                          actionButton("rf_conf_toggle", 
                                      ifelse(TRUE, "▼ Show explanation", "▲ Hide explanation"), 
                                      class = "toggle-btn")
                        ),
                        verbatimTextOutput("rf_conf_matrix"),
                        conditionalPanel(
                          condition = "input.rf_conf_toggle % 2 == 1",
                          div(class = "explanation-box",
                            tags$ul(
                              tags$li("Rows: What the model predicted (0=Benign, 1=Malignant)"),
                              tags$li("Columns: Actual diagnosis (0=Benign, 1=Malignant)"),
                              tags$li("Diagonal (top-left to bottom-right): Correct predictions"),
                              tags$li("Off-diagonal: Errors (missed diagnoses)")
                            )
                          )
                        )
                      ),
                      div(style = "width: 48%; min-width: 250px; background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                        div(style = "display: flex; align-items: center;",
                          h4("Performance Metrics", align = "center", style = "margin-right: auto;"),
                          actionButton("rf_metrics_toggle", 
                                      ifelse(TRUE, "▼ Show explanation", "▲ Hide explanation"), 
                                      class = "toggle-btn")
                        ),
                        verbatimTextOutput("rf_performance_metrics"),
                        conditionalPanel(
                          condition = "input.rf_metrics_toggle % 2 == 1",
                          div(class = "explanation-box",
                            tags$ul(
                              tags$li(strong("Accuracy:"), "Percentage of all correct predictions"),
                              tags$li(strong("Precision:"), "When model predicts malignant, how often it's right"),
                              tags$li(strong("Recall:"), "Percentage of actual malignant tumors found"),
                              tags$li(strong("Specificity:"), "Percentage of actual benign tumors correctly identified"),
                              tags$li(strong("F1 Score:"), "Balance between precision and recall (1 is best)")
                            )
                          )
                        )
                      )
                    )
                  )
                ),
                # variable importance, similar to decision tree
                tabPanel("Variable Importance", 
                  div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                    h4("Feature Importance Explained", align = "center"),
                    p("This chart shows which measurements are most useful for diagnosis. Longer bars mean that feature is more important for predicting malignancy.")
                  ),
                  plotOutput("rf_varImpPlot")
                ),
                # prediction tool similar to decision tree
                tabPanel("Prediction Tool",
                  sidebarLayout(
                    sidebarPanel(
                      div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                        h4("Enter Patient Data", align = "center"),
                        do.call(tabsetPanel, c(
                          list(id = "rf_featureTabs"),
                          lapply(seq_along(predictor_groups), function(i) {
                            group <- predictor_groups[[i]]
                            tabPanel(
                              title = paste("Features", ((i - 1) * 5 + 1), "to", i * 5),
                              do.call(tagList, lapply(group, function(col) {
                                sliderInput(inputId = paste0("rf_", col), label = col,
                                            min = min(cancer[[col]], na.rm = TRUE),
                                            max = max(cancer[[col]], na.rm = TRUE),
                                            value = median(cancer[[col]], na.rm = TRUE),
                                            step = (max(cancer[[col]], na.rm = TRUE) - min(cancer[[col]], na.rm = TRUE)) / 100)
                              }))
                            )
                          })
                        )),
                        actionButton("rf_predict_btn", "Predict", class = "btn-primary", style = "width: 100%; margin-top: 10px;")
                      )
                    ),
                    mainPanel(
                      div(style = "background-color: #f8f9fa; padding: 15px; border-radius: 10px; margin-bottom: 20px;",
                        h4("Prediction Result", align = "center"), 
                        verbatimTextOutput("rf_prediction_result"),
                        plotlyOutput("rf_prediction_prob_plot")
                      )
                    )
                  )
                )
              )
            )
          )
        )
  )
)

server <- function(input, output, session) {
  
  # toggle performance explanations
  observeEvent(input$dt_conf_toggle, {
    if (input$dt_conf_toggle %% 2 == 1) {
      updateActionButton(session, "dt_conf_toggle", "▲ Hide explanation")
    } else {
      updateActionButton(session, "dt_conf_toggle", "▼ Show explanation")
    }
  })
  
  # toggle performance explanations
  observeEvent(input$dt_metrics_toggle, {
    if (input$dt_metrics_toggle %% 2 == 1) {
      updateActionButton(session, "dt_metrics_toggle", "▲ Hide explanation")
    } else {
      updateActionButton(session, "dt_metrics_toggle", "▼ Show explanation")
    }
  })
  
  # toggle performance explanations
  observeEvent(input$rf_conf_toggle, {
    if (input$rf_conf_toggle %% 2 == 1) {
      updateActionButton(session, "rf_conf_toggle", "▲ Hide explanation")
    } else {
      updateActionButton(session, "rf_conf_toggle", "▼ Show explanation")
    }
  })
  
  # toggle performance explanations
  observeEvent(input$rf_metrics_toggle, {
    if (input$rf_metrics_toggle %% 2 == 1) {
      updateActionButton(session, "rf_metrics_toggle", "▲ Hide explanation")
    } else {
      updateActionButton(session, "rf_metrics_toggle", "▼ Show explanation")
    }
  })
  
  # ---------------------
  # Decision Tree Model
  # ---------------------

  # create model and maken it reactive to input
  modelFit <- reactive({
    rpart(diagnosis ~ ., data = train_data, method = "class",
          parms = list(split = input$splitCriteria),
          control = rpart.control(maxdepth = input$maxdepth, 
                                  cp = input$cp, 
                                  minsplit = input$minsplit))
  })
  
  # create output for tree plot
  output$treePlot <- renderPlot({
    rpart.plot(modelFit())
  })
  
  # create output for roc curve
  output$rocPlot <- renderPlotly({
    test_pred_prob <- predict(modelFit(), test_data, type = "prob")[, "1"]
    binary_preds <- ifelse(test_pred_prob >= input$threshold, 1, 0)
    
    # load output for conf mat
    conf_mat <- table(Predicted = binary_preds, Actual = test_data$diagnosis)
    tp <- if("1" %in% rownames(conf_mat) && "1" %in% colnames(conf_mat)) conf_mat["1", "1"] else 0
    tn <- if("0" %in% rownames(conf_mat) && "0" %in% colnames(conf_mat)) conf_mat["0", "0"] else 0
    fp <- if("1" %in% rownames(conf_mat) && "0" %in% colnames(conf_mat)) conf_mat["1", "0"] else 0
    fn <- if("0" %in% rownames(conf_mat) && "1" %in% colnames(conf_mat)) conf_mat["0", "1"] else 0
    sens <- if((tp + fn) > 0) tp / (tp + fn) else 0
    spec <- if((tn + fp) > 0) tn / (tn + fp) else 0
    
    roc_df <- data.frame(
      fpr = c(0, 1 - spec, 1),
      tpr = c(0, sens, 1)
    )
    
    # create auc value for plot
    auc_value <- with(roc_df, sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1))) / 2)
    
    # create plot for ROC incl threshold and auc
    plot_ly(roc_df, x = ~fpr, y = ~tpr, type = 'scatter', mode = 'lines+markers',
            line = list(color = "#3498db"), marker = list(color = "#3498db"),
            name = paste("Threshold =", input$threshold)) %>%
      layout(title = paste("ROC Curve (Threshold =", input$threshold, ", AUC =", round(auc_value, 3), ")"),
             xaxis = list(title = "False Positive Rate"),
             yaxis = list(title = "True Positive Rate"))
  })
  
  # output for conf mat
  output$conf_matrix <- renderPrint({
    test_pred_prob <- predict(modelFit(), newdata = test_data, type = "prob")[, "1"]
    pred_class <- factor(ifelse(test_pred_prob >= input$threshold, 1, 0), levels = c(0, 1))
    conf_mat <- table(Predicted = pred_class, Actual = test_data$diagnosis)
    print(conf_mat)
    invisible()
  })
  
  # create performance metrics output
  output$performance_metrics <- renderPrint({
    test_pred_prob <- predict(modelFit(), newdata = test_data, type = "prob")[, "1"]
    pred_class <- factor(ifelse(test_pred_prob >= input$threshold, 1, 0), levels = c(0,1))
    conf_mat <- table(Predicted = pred_class, Actual = test_data$diagnosis)
    
    tp <- if("1" %in% rownames(conf_mat) && "1" %in% colnames(conf_mat)) conf_mat["1", "1"] else 0
    tn <- if("0" %in% rownames(conf_mat) && "0" %in% colnames(conf_mat)) conf_mat["0", "0"] else 0
    fp <- if("1" %in% rownames(conf_mat) && "0" %in% colnames(conf_mat)) conf_mat["1", "0"] else 0
    fn <- if("0" %in% rownames(conf_mat) && "1" %in% colnames(conf_mat)) conf_mat["0", "1"] else 0
    
    accuracy <- (tp + tn) / (tp + tn + fp + fn)
    precision <- if((tp + fp) > 0) tp / (tp + fp) else NA
    recall <- if((tp + fn) > 0) tp / (tp + fn) else NA
    specificity <- if((tn + fp) > 0) tn / (tn + fp) else NA
    f1 <- if(!is.na(precision) && !is.na(recall) && (precision + recall) > 0) 2 * precision * recall / (precision + recall) else NA
    
    cat("Threshold:", round(input$threshold, 2), "\n")
    cat("Accuracy:", round(accuracy, 4), "\n")
    cat("Precision:", round(precision, 4), "\n")
    cat("Recall (Sensitivity):", round(recall, 4), "\n")
    cat("Specificity:", round(specificity, 4), "\n")
    cat("F1 Score:", round(f1, 4), "\n")
    invisible()
  })
  
  # create variable importance
  output$varImpPlot <- renderPlot({
    importance <- varImp(modelFit(), scale = FALSE)
    importance_df <- as.data.frame(importance$Overall)
    importance_df$Overall <- importance$Overall
    importance_df$Variable <- rownames(importance)
    importance_df <- importance_df[, c("Variable", "Overall")]
    importance_df <- importance_df[importance_df$Overall > 0, ]
    ggplot(importance_df, aes(x = reorder(Variable, Overall), y = Overall)) +
      geom_bar(stat = "identity", fill = "#2c3e50") +
      coord_flip() +
      labs(title = "Variable Importance", x = "Features", y = "Importance Score") +
      theme_minimal()
  })
  
  # perdition tool button
  predResult <- eventReactive(input$predict_btn, {
    new_obs <- as.data.frame(as.list(sapply(predictor_cols, function(col) as.numeric(input[[col]]))))
    pred_class <- predict(modelFit(), newdata = new_obs, type = "class")
    pred_prob <- predict(modelFit(), newdata = new_obs, type = "prob")
    list(class = pred_class, prob = pred_prob)
  })
  
  # prediction output box text
  output$prediction_result <- renderPrint({
    res <- predResult()
    cat("Predicted Diagnosis:", as.character(res$class))
  })
  
  # create plot with perdiction porbabilities
  output$prediction_prob_plot <- renderPlotly({
    res <- predResult()
    prob_df <- data.frame(
      Diagnosis = names(res$prob[1,]),
      Probability = as.numeric(res$prob[1,])
    )
    colors <- c("#2ecc71", "#e74c3c")  # Benign (0), Malignant (1)
    
    plot_ly(prob_df, x = ~Diagnosis, y = ~Probability, type = 'bar',
            marker = list(color = colors)) %>%
      layout(title = "Prediction Probabilities",
             xaxis = list(title = "Diagnosis"),
             yaxis = list(title = "Probability"))
  })
  
  # --------------------------
  # Random Forest Model Setup
  # --------------------------

  # create model and maken it reactive to input
  rfModelFit <- reactive({
    randomForest(as.factor(diagnosis) ~ ., data = train_data,
                 ntree = input$rf_ntree,
                 mtry = input$rf_mtry,
                 nodesize = input$rf_nodesize,
                 replace = input$rf_replace)
  })
  
  # create variable importance
  output$rf_varImpPlot <- renderPlot({
    varImp_rf <- importance(rfModelFit())
    varImp_df <- data.frame(Variable = rownames(varImp_rf), Importance = varImp_rf[, 1])
    ggplot(varImp_df, aes(x = reorder(Variable, Importance), y = Importance)) +
      geom_bar(stat = "identity", fill = "#2c3e50") +
      coord_flip() +
      labs(title = "Random Forest Variable Importance", x = "Features", y = "Importance") +
      theme_minimal()
  })
  
  # create metrics for roc cruve and plot
  output$rf_rocPlot <- renderPlotly({
    test_pred_prob <- predict(rfModelFit(), test_data, type = "prob")[, "1"]
    binary_preds <- ifelse(test_pred_prob >= input$rf_threshold, 1, 0)
    
    conf_mat <- table(Predicted = binary_preds, Actual = test_data$diagnosis)
    tp <- if("1" %in% rownames(conf_mat) && "1" %in% colnames(conf_mat)) conf_mat["1", "1"] else 0
    tn <- if("0" %in% rownames(conf_mat) && "0" %in% colnames(conf_mat)) conf_mat["0", "0"] else 0
    fp <- if("1" %in% rownames(conf_mat) && "0" %in% colnames(conf_mat)) conf_mat["1", "0"] else 0
    fn <- if("0" %in% rownames(conf_mat) && "1" %in% colnames(conf_mat)) conf_mat["0", "1"] else 0
    sens <- if((tp + fn) > 0) tp / (tp + fn) else 0
    spec <- if((tn + fp) > 0) tn / (tn + fp) else 0
    
    roc_df <- data.frame(
      fpr = c(0, 1 - spec, 1),
      tpr = c(0, sens, 1)
    )

    # create auc value
    auc_value <- with(roc_df, sum(diff(fpr) * (head(tpr, -1) + tail(tpr, -1))) / 2)
    
    # plot the curve
    plot_ly(roc_df, x = ~fpr, y = ~tpr, type = 'scatter', mode = 'lines+markers',
            line = list(color = "#3498db"), marker = list(color = "#3498db"),
            name = paste("Threshold =", input$rf_threshold)) %>%
      layout(title = paste("RF ROC Curve (Threshold =", input$rf_threshold, ", AUC =", round(auc_value, 3), ")"),
             xaxis = list(title = "False Positive Rate"),
             yaxis = list(title = "True Positive Rate"))
  })
  
  # print conf mat
  output$rf_conf_matrix <- renderPrint({
    test_pred_prob <- predict(rfModelFit(), newdata = test_data, type = "prob")[, "1"]
    pred_class <- factor(ifelse(test_pred_prob >= input$rf_threshold, 1, 0), levels = c(0, 1))
    conf_mat <- table(Predicted = pred_class, Actual = test_data$diagnosis)
    print(conf_mat)
    invisible()
  })
  
  # calculate and print performance metrics like accuracy etc.
  output$rf_performance_metrics <- renderPrint({
    test_pred_prob <- predict(rfModelFit(), newdata = test_data, type = "prob")[, "1"]
    pred_class <- factor(ifelse(test_pred_prob >= input$rf_threshold, 1, 0), levels = c(0,1))
    conf_mat <- table(Predicted = pred_class, Actual = test_data$diagnosis)
    
    tp <- if("1" %in% rownames(conf_mat) && "1" %in% colnames(conf_mat)) conf_mat["1", "1"] else 0
    tn <- if("0" %in% rownames(conf_mat) && "0" %in% colnames(conf_mat)) conf_mat["0", "0"] else 0
    fp <- if("1" %in% rownames(conf_mat) && "0" %in% colnames(conf_mat)) conf_mat["1", "0"] else 0
    fn <- if("0" %in% rownames(conf_mat) && "1" %in% colnames(conf_mat)) conf_mat["0", "1"] else 0
    
    accuracy <- (tp + tn) / (tp + tn + fp + fn)
    precision <- if((tp + fp) > 0) tp / (tp + fp) else NA
    recall <- if((tp + fn) > 0) tp / (tp + fn) else NA
    specificity <- if((tn + fp) > 0) tn / (tn + fp) else NA
    f1 <- if(!is.na(precision) && !is.na(recall) && (precision + recall) > 0) 2 * precision * recall / (precision + recall) else NA
    
    cat("RF Threshold:", round(input$rf_threshold, 2), "\n")
    cat("Accuracy:", round(accuracy, 4), "\n")
    cat("Precision:", round(precision, 4), "\n")
    cat("Recall (Sensitivity):", round(recall, 4), "\n")
    cat("Specificity:", round(specificity, 4), "\n")
    cat("F1 Score:", round(f1, 4), "\n")
    invisible()
  })
  
  # perdition tool button
  rfPredResult <- eventReactive(input$rf_predict_btn, {
    new_obs <- as.data.frame(as.list(sapply(predictor_cols, function(col) as.numeric(input[[paste0("rf_", col)]]))))
    pred_class <- predict(rfModelFit(), newdata = new_obs, type = "class")
    pred_prob <- predict(rfModelFit(), newdata = new_obs, type = "prob")
    list(class = pred_class, prob = pred_prob)
  })
  
  # prediction output box text
  output$rf_prediction_result <- renderPrint({
    res <- rfPredResult()
    cat("Predicted Diagnosis:", as.character(res$class))
  })
  
  # create plot with perdiction porbabilities
  output$rf_prediction_prob_plot <- renderPlotly({
    res <- rfPredResult()
    prob_df <- data.frame(
      Diagnosis = names(res$prob[1,]),
      Probability = as.numeric(res$prob[1,])
    )
    colors <- c("#2ecc71", "#e74c3c")  # Benign (0), Malignant (1)
    
    plot_ly(prob_df, x = ~Diagnosis, y = ~Probability, type = 'bar',
            marker = list(color = colors)) %>%
      layout(title = "RF Prediction Probabilities",
             xaxis = list(title = "Diagnosis"),
             yaxis = list(title = "Probability"))
  })
}

# Run app
shinyApp(ui = ui, server = server)
