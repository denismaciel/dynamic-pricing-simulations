library(tidyverse)

df <- list.files('data', pattern = '*.csv', full.names = TRUE) %>%
  map(function(x) read_csv(x) %>% mutate(experiment = x)) %>%
  bind_rows

CLAIRVOYANT_REVENUE_025 <- (0.75 * 0.3 * 39.9) + (0.25 * 0.1 * 44.9)
CLAIRVOYANT_REVENUE_050 <- (2/3 * 0.6 * 34.9) + (1/3 * 0.3 * 39.9)


# Sanity check
df %>%
  group_by(experiment, id) %>%
  summarise(n_periods = max(step) + 1, n = n()) %>%
  group_by(experiment) %>%
  summarise(mean(n_periods))


# Cleaner approach
trial_names <- tibble(
  trial_factory = c(
    "ts_fixed_with_bernoulli",
    "ts_update_with_bernoulli",
    "ts_ignore_inventory_with_bernoulli"
  ),
  algorithm = c("TS-Fixed", "TS-Update", "TS")
)

clairvoyant_revenue = tibble(
  inventory = c("0.25.", "0.5."),
  clairvoyant_revenue = c(CLAIRVOYANT_REVENUE_025, CLAIRVOYANT_REVENUE_050)
)

experiment_from_file_name <- function(file_name) {
  str_extract(file_name, "ts_fixed_with_bernoulli|ts_update_with_bernoulli|ts_ignore_inventory_with_bernoulli")
}

inventory_from_file_name <- function(file_name) {
  str_extract(file_name, "(?<=alpha)[0-9\\.]+")
}

nperiods_from_file_name <- function(file_name) {
  as.integer(str_extract(file_name, "\\d{3,}"))
}

experiment_from_file_name("data/ts_fixed_with_bernoulli_.csv")
experiment_from_file_name("data/ts_update_with_bernoulli_.csv")
experiment_from_file_name("data/ts_ignore_inventory_with_bernoulli_.csv")

assertthat::assert_that(inventory_from_file_name('alpha0.5.csv') == "0.5.")
assertthat::assert_that(inventory_from_file_name('alpha0.25.csv') == "0.25.")
  
avg_revenue <- df %>%
  group_by(experiment, trial_id = id) %>%
  summarise(total_revenue = sum(price * demand, na.rm = TRUE)) %>%
  mutate(
    n_periods = nperiods_from_file_name(experiment),
    trial_factory = experiment_from_file_name(experiment),
    inventory = inventory_from_file_name(experiment)
  ) %>%
  left_join(trial_names) %>%
  left_join(clairvoyant_revenue) %>%
  mutate(avg_revenue_per_step = total_revenue / n_periods) %>%
  group_by(algorithm, n_periods, inventory) %>%
  summarise(as_pp_of_clairvoyant = mean(avg_revenue_per_step,  na.rm = TRUE) / clairvoyant_revenue)

x_axis <-  tibble(labels = unique(avg_revenue$n_periods)) %>%
  mutate(breaks = log(labels))

plot <- avg_revenue %>%
  ggplot(aes(
    log(n_periods),
    as_pp_of_clairvoyant,
    color = algorithm,
    shape = algorithm
  )) +
  geom_point() +
  geom_line() +
  labs(x = "Number of Periods T (in log scale)",
       y = NULL,
       title = "Percent of Optimal Revenue Achieved") +
  theme_minimal() +
  scale_y_continuous(labels = scales::percent,
                     breaks = scales::pretty_breaks(n = 8)) +
  scale_x_continuous(labels = x_axis$labels, breaks = x_axis$breaks) +
  theme(
    legend.title = element_blank(),
    legend.position = "bottom",
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.minor.y = element_blank(),
  ) +
  facet_wrap(~inventory)
plot

ggsave(fs::path('figs', 'replication.png'), plot, width = 9, height = 6)
