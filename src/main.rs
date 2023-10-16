// Code to solve Rock Paper Scissors using Counterfactual Regret Minimization
// Following "An Introduction to Counterfactual Regret Minimization" by Neller and Lanctot (2013)

use rand::Rng;
use std::env;

// Things I'd change for an actual release:
// don't put everything in main
// add documentation
// add tests

fn main() {
    let num_iters = read_num_iters();

    let mut rng = rand::thread_rng();

    let mut my_strategy_sum = new_vals();
    let mut my_regret_sum = new_vals();
    let mut opp_strategy_sum = new_vals();
    let mut opp_regret_sum = new_vals();

    train_strategy(
        num_iters,
        &mut my_strategy_sum,
        &mut my_regret_sum,
        &mut opp_strategy_sum,
        &mut opp_regret_sum,
        &mut rng,
    );

    let my_avg_strategy = get_average_strategy(&my_strategy_sum);
    println!("My average strategy is {:?}", my_avg_strategy);

    let opp_avg_strategy = get_average_strategy(&opp_strategy_sum);
    println!("Opp average strategy is {:?}", opp_avg_strategy);
}

#[derive(Copy, Clone, PartialEq, Eq)]
enum RpsThrow {
    Rock,
    Paper,
    Scissors,
    Water,
    Fire,
}
const THROW_LIST: [RpsThrow; 5] = [
    RpsThrow::Rock,
    RpsThrow::Paper,
    RpsThrow::Scissors,
    RpsThrow::Water,
    RpsThrow::Fire,
];

fn game_value(my_throw: &RpsThrow, opp_throw: &RpsThrow) -> f64 {
    let rps_beats_list = [
        (RpsThrow::Rock, RpsThrow::Scissors),
        (RpsThrow::Scissors, RpsThrow::Paper),
        (RpsThrow::Paper, RpsThrow::Rock),
    ];
    if *my_throw == *opp_throw {
        return 0.0;
    } else if rps_beats_list.contains(&(*my_throw, *opp_throw)) {
        return 1.0;
    } else if rps_beats_list.contains(&(*opp_throw, *my_throw)) {
        return -1.0;
    } else if *my_throw == RpsThrow::Fire {
        return if *opp_throw == RpsThrow::Water {
            -1.0
        } else {
            1.0
        };
    } else if *my_throw == RpsThrow::Water {
        return if *opp_throw == RpsThrow::Fire {
            1.0
        } else {
            -1.0
        };
    } else if *opp_throw == RpsThrow::Fire {
        assert!([RpsThrow::Rock, RpsThrow::Paper, RpsThrow::Scissors].contains(&my_throw));
        return -1.0;
    } else if *opp_throw == RpsThrow::Water {
        assert!([RpsThrow::Rock, RpsThrow::Paper, RpsThrow::Scissors].contains(&my_throw));
        return 1.0;
    } else {
        assert!(false, "Previous match should have been exhaustive");
        return 0.0;
    }
}

#[derive(Debug)]
struct ThrowVals {
    rock: f64,
    paper: f64,
    scissors: f64,
    water: f64,
    fire: f64,
}

impl ThrowVals {
    fn get_val(&self, throw: &RpsThrow) -> f64 {
        match throw {
            RpsThrow::Rock => self.rock,
            RpsThrow::Paper => self.paper,
            RpsThrow::Scissors => self.scissors,
            RpsThrow::Water => self.water,
            RpsThrow::Fire => self.fire,
        }
    }

    fn write_val(&mut self, throw: &RpsThrow) -> &mut f64 {
        match throw {
            RpsThrow::Rock => &mut self.rock,
            RpsThrow::Paper => &mut self.paper,
            RpsThrow::Scissors => &mut self.scissors,
            RpsThrow::Water => &mut self.water,
            RpsThrow::Fire => &mut self.fire,
        }
    }
}

fn new_vals() -> ThrowVals {
    ThrowVals {
        rock: 0.0,
        paper: 0.0,
        scissors: 0.0,
        water: 0.0,
        fire: 0.0,
    }
}

fn get_strategy(regret_sum: &ThrowVals, strategy_sum: &mut ThrowVals) -> ThrowVals {
    let mut strategy = new_vals();
    let mut normalizing_sum = 0.0;

    for throw in THROW_LIST {
        let r = regret_sum.get_val(&throw);
        let throw_strat = if r > 0.0 { r } else { 0.0 };
        *strategy.write_val(&throw) = throw_strat;
        normalizing_sum += throw_strat;
    }

    for throw in THROW_LIST {
        let throw_strat = strategy.get_val(&throw);
        let new_strat = if normalizing_sum > 0.0 {
            throw_strat / normalizing_sum
        } else {
            1.0 / (THROW_LIST.len() as f64)
        };
        *strategy.write_val(&throw) = new_strat;
        *strategy_sum.write_val(&throw) += new_strat;
    }

    strategy
}

fn sample_strategy(strategy: &ThrowVals, rng: &mut rand::rngs::ThreadRng) -> RpsThrow {
    // TODO: somehow check that weights approximately add to one...
    let r: f64 = rng.gen();
    let mut cum_prob = 0.0;
    for throw in THROW_LIST {
        cum_prob += strategy.get_val(&throw);
        if cum_prob > r {
            return throw;
        }
    }
    THROW_LIST[THROW_LIST.len() - 1]
}

fn get_action_pair(
    my_regret_sum: &ThrowVals,
    my_strategy_sum: &mut ThrowVals,
    opp_regret_sum: &ThrowVals,
    opp_strategy_sum: &mut ThrowVals,
    rng: &mut rand::rngs::ThreadRng,
) -> (RpsThrow, RpsThrow) {
    let my_strategy = get_strategy(my_regret_sum, my_strategy_sum);
    let my_action = sample_strategy(&my_strategy, rng);
    let opp_strategy = get_strategy(opp_regret_sum, opp_strategy_sum);
    let opp_action = sample_strategy(&opp_strategy, rng);
    (my_action, opp_action)
}

fn best_response_utilities(opp_action: &RpsThrow) -> ThrowVals {
    let mut utilities = new_vals();
    for throw in THROW_LIST {
        *utilities.write_val(&throw) = game_value(&throw, opp_action);
    }
    utilities
}

fn accumulate_regrets(action: &RpsThrow, utilities: &ThrowVals, regret_sum: &mut ThrowVals) {
    for throw in THROW_LIST {
        *regret_sum.write_val(&throw) += utilities.get_val(&throw) - utilities.get_val(action);
    }
}

fn train_strategy(
    num_iters: i32,
    my_strategy_sum: &mut ThrowVals,
    my_regret_sum: &mut ThrowVals,
    opp_strategy_sum: &mut ThrowVals,
    opp_regret_sum: &mut ThrowVals,
    rng: &mut rand::rngs::ThreadRng,
) {
    for _i in 0..num_iters {
        let (my_action, opp_action) = get_action_pair(
            my_regret_sum,
            my_strategy_sum,
            opp_regret_sum,
            opp_strategy_sum,
            rng,
        );
        let my_utils = best_response_utilities(&opp_action);
        accumulate_regrets(&my_action, &my_utils, my_regret_sum);
        let opp_utils = best_response_utilities(&my_action);
        accumulate_regrets(&opp_action, &opp_utils, opp_regret_sum);
    }
}

fn get_average_strategy(strategy_sum: &ThrowVals) -> ThrowVals {
    let mut avg_strat = new_vals();
    let mut normalizing_sum = 0.0;
    for throw in THROW_LIST {
        normalizing_sum += strategy_sum.get_val(&throw);
    }
    for throw in THROW_LIST {
        *avg_strat.write_val(&throw) = if normalizing_sum > 0.0 {
            strategy_sum.get_val(&throw) / normalizing_sum
        } else {
            1.0 / (THROW_LIST.len() as f64)
        };
    }
    avg_strat
}

fn read_num_iters() -> i32 {
    let mut args_iter = env::args();
    args_iter.next();
    let num_iters_string: Option<String> = args_iter.next();
    match num_iters_string {
        Some(s) => {
            let num_iters_result = s.parse::<i32>();
            match num_iters_result {
                Ok(x) => {
                    println!("Running with {} iterations", x);
                    x
                }
                Err(e) => {
                    println!("Parsing input for number of iterations failed: {}", e);
                    println!("Using default value of 100,000");
                    100_000
                }
            }
        }
        None => {
            println!("No value for number of iterations provided. Using default value of 100,000.");
            100_000
        }
    }
}
