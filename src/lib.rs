#![recursion_limit = "1024"]

use glenside::language::interpreter::Environment;
use monaco::{
    api::CodeEditorOptions,
    sys::editor::BuiltinTheme,
    yew::{CodeEditor, CodeEditorLink},
};
use ndarray::{ArrayD, Dimension};
use rand::{
    distributions::{Distribution, Uniform},
    rngs::OsRng,
};
use std::rc::Rc;
use wasm_bindgen::prelude::*;
use yew::{html, html_nested, Component, ComponentLink, Html, InputData, Properties, ShouldRender};

fn get_options() -> CodeEditorOptions {
    CodeEditorOptions::default()
        .with_new_dimension(500, 500)
        .with_builtin_theme(BuiltinTheme::VsDark)
}

enum Message {
    NewInput,
    EnvironmentValueUpdated(String, ArrayD<f64>),
    AddNewEnvironmentInput,
}

struct App {
    options: Rc<CodeEditorOptions>,
    link: ComponentLink<Self>,
    code_editor_link: CodeEditorLink,
    result_text: String,
    environment: Environment<'static, f64>,
    num_environment_inputs: usize,
}
impl Component for App {
    type Message = Message;
    type Properties = ();

    fn create(_props: Self::Properties, link: ComponentLink<Self>) -> Self {
        Self {
            options: Rc::new(get_options()),
            link: link,
            code_editor_link: CodeEditorLink::default(),
            result_text: String::default(),
            environment: Environment::default(),
            num_environment_inputs: 0,
        }
    }

    fn update(&mut self, msg: Self::Message) -> ShouldRender {
        match msg {
            Message::AddNewEnvironmentInput => {
                self.num_environment_inputs += 1;
                true
            }
            Message::EnvironmentValueUpdated(name, value) => {
                let name = Box::leak(name.into_boxed_str());
                self.environment.insert(name, value);
                false
            }
            Message::NewInput => {
                let text_input = self
                    .code_editor_link
                    .with_editor(|editor| editor.get_model().unwrap().get_value())
                    .unwrap();

                let result = glenside::language::interpreter::interpret_from_str::<f64>(
                    &text_input,
                    &self.environment,
                );

                let text_output = match result {
                    glenside::language::interpreter::Value::Tensor(t) => t.to_string(),
                    glenside::language::interpreter::Value::Access(a) => {
                        format!(
                            "shape: (({a}), ({b}))\n\
                             value:\n\
                             {tensor}",
                            a = a.tensor.shape()[..a.access_axis]
                                .iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<_>>()
                                .join(", "),
                            b = a.tensor.shape()[a.access_axis..]
                                .iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<_>>()
                                .join(", "),
                            tensor = a.tensor.to_string()
                        )
                    }
                    glenside::language::interpreter::Value::Usize(_) => todo!(),
                    glenside::language::interpreter::Value::Shape(_) => todo!(),
                    glenside::language::interpreter::Value::ComputeType(_) => todo!(),
                    glenside::language::interpreter::Value::PadType(_) => todo!(),
                    glenside::language::interpreter::Value::AccessShape(shape, access_axis) => {
                        format!(
                            "(({a}), ({b}))",
                            a = shape.slice()[..access_axis]
                                .iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<_>>()
                                .join(", "),
                            b = shape.slice()[access_axis..]
                                .iter()
                                .map(|i| i.to_string())
                                .collect::<Vec<_>>()
                                .join(", ")
                        )
                    }
                    glenside::language::interpreter::Value::List(_) => todo!(),
                };

                self.result_text = text_output;

                true
            }
        }
    }

    fn change(&mut self, _props: Self::Properties) -> ShouldRender {
        false
    }

    fn view(&self) -> Html {
        html! {
            <div>
                <input type={"button"} value={"+"} onclick=self.link.callback(|_| Message::AddNewEnvironmentInput) />
                {
                    for (0..self.num_environment_inputs).map(|i| {
                        html_nested!{
                            <GeneratedTensorEnvironmentInput
                                id={i}
                                value_updated_callback=self.link.callback(|(name, value)| {
                                    Message::EnvironmentValueUpdated(name, value)
                                }) />
                        }
                    })
                }
                <CodeEditor link=&self.code_editor_link, options=Rc::clone(&self.options) />
                <input type={"button"} value={"run"} onclick=self.link.callback(|_| Message::NewInput) />
                <textarea readonly={true}>{self.result_text.clone()}</textarea>
            </div>
        }
    }
}

#[derive(Properties, Clone)]
struct EnvironmentInputProps {
    value_updated_callback: yew::Callback<(String, ArrayD<f64>)>,
    /// Unique id identifying this input in a list of inputs. Currently only
    /// used so that we can make the names of the radio button groups unique.
    id: usize,
}

struct GeneratedTensorEnvironmentInput {
    properties: EnvironmentInputProps,
    link: ComponentLink<Self>,
    name: String,
    shape_string: String,
    value_generation_strategy: Option<ValueGenerationStrategy>,
}

impl GeneratedTensorEnvironmentInput {
    fn get_value(&self) -> Option<(String, ArrayD<f64>)> {
        // First and last characters should be parens.
        if self.shape_string.is_empty()
            || self.shape_string.chars().nth(0).unwrap() != '('
            || self
                .shape_string
                .chars()
                .nth(self.shape_string.len() - 1)
                .unwrap()
                != ')'
        {
            return None;
        }

        let parens_trimmed = &self.shape_string[1..self.shape_string.len() - 1];

        let parse_results = if parens_trimmed.is_empty() {
            vec![]
        } else {
            parens_trimmed
                .split(",")
                .map(|s| s.parse::<usize>())
                .collect::<Vec<_>>()
        };

        if !parse_results.is_empty() && parse_results.iter().any(|r| r.is_err()) {
            return None;
        }

        let shape = parse_results
            .iter()
            .map(|r| *r.as_ref().unwrap())
            .collect::<Vec<_>>();

        match self.value_generation_strategy {
            Some(ValueGenerationStrategy::Zeros) => {
                Some((self.name.clone(), ndarray::ArrayD::zeros(shape)))
            }
            Some(ValueGenerationStrategy::Ones) => {
                Some((self.name.clone(), ndarray::ArrayD::ones(shape)))
            }
            Some(ValueGenerationStrategy::Random) => Some((
                self.name.clone(),
                ndarray::ArrayD::from_shape_fn(shape, |_| {
                    Uniform::new(-2.0, 2.0).sample(&mut OsRng::new().unwrap())
                }),
            )),
            None => None,
        }
    }
}
enum ValueGenerationStrategy {
    Random,
    Zeros,
    Ones,
}

enum GeneratedTensorEnvironmentInputMessage {
    UpdateName(String),
    UpdateShapeString(String),
    UpdateValueGenerationStrategy(ValueGenerationStrategy),
}

impl Component for GeneratedTensorEnvironmentInput {
    type Message = GeneratedTensorEnvironmentInputMessage;
    type Properties = EnvironmentInputProps;

    fn create(properties: Self::Properties, link: ComponentLink<Self>) -> Self {
        Self {
            properties,
            link,
            name: String::default(),
            shape_string: String::default(),
            value_generation_strategy: None,
        }
    }

    fn update(&mut self, msg: Self::Message) -> ShouldRender {
        match msg {
            GeneratedTensorEnvironmentInputMessage::UpdateName(s) => self.name = s,
            GeneratedTensorEnvironmentInputMessage::UpdateShapeString(s) => self.shape_string = s,
            GeneratedTensorEnvironmentInputMessage::UpdateValueGenerationStrategy(s) => {
                self.value_generation_strategy = Some(s)
            }
        }

        if let Some(value) = self.get_value() {
            self.properties.value_updated_callback.emit(value);
        }

        true
    }

    fn change(&mut self, properties: Self::Properties) -> ShouldRender {
        self.properties = properties;
        true
    }

    fn view(&self) -> Html {
        html! {
            <div>
                // Name text box
                <label for={"name"}>{"Name"}</label>
                <input name={"name"} type={"text"} oninput=self.link.callback(
                    |event: InputData| GeneratedTensorEnvironmentInputMessage::UpdateName(event.value)) />

                // Shape text box
                <label for={"shape"}>{"Shape (e.g. () or (3,32,32))"}</label>
                <input name={"shape"} type={"text"}
                    oninput=self.link.callback(|event: InputData| {
                        GeneratedTensorEnvironmentInputMessage::UpdateShapeString(event.value)
                    })
                />

                // Value generation radio buttons
                <input type={"radio"}
                    id={format!("random-{}", self.properties.id)}
                    name={format!("values-{}", self.properties.id)}
                    checked={match self.value_generation_strategy {
                        Some(ValueGenerationStrategy::Random) => true,
                        _ => false,
                    }}
                    oninput=self.link.callback(|_|
                        GeneratedTensorEnvironmentInputMessage::UpdateValueGenerationStrategy(
                            ValueGenerationStrategy::Random
                        ))
                />
                <label for={format!("random-{}", self.properties.id)}>{"random"}</label>

                <input type={"radio"}
                    id={format!("zeros-{}", self.properties.id)}
                    name={format!("values-{}", self.properties.id)}
                    checked={match self.value_generation_strategy {
                        Some(ValueGenerationStrategy::Zeros) => true,
                        _ => false,
                    }}
                    oninput=self.link.callback(|_|
                        GeneratedTensorEnvironmentInputMessage::UpdateValueGenerationStrategy(
                            ValueGenerationStrategy::Zeros
                        ))
                />
                <label for={format!("zeros-{}", self.properties.id)}>{"zeros"}</label>

                <input type={"radio"}
                    id={format!("ones-{}", self.properties.id)}
                    name={format!("values-{}", self.properties.id)}
                    checked={match self.value_generation_strategy {
                        Some(ValueGenerationStrategy::Ones) => true,
                        _ => false,
                    }}
                    oninput=self.link.callback(|_|
                        GeneratedTensorEnvironmentInputMessage::UpdateValueGenerationStrategy(
                            ValueGenerationStrategy::Ones
                        ))
                />
                <label for={format!("ones-{}", self.properties.id)}>{"ones"}</label>

                <input
                    type={"checkbox"}
                    id={"valid"}
                    disabled={true}
                    checked={match self.get_value() { Some(_) => true, _ => false}}
                />
                <label for={"valid"}>{"valid?"}</label>
            </div>
        }
    }
}

#[wasm_bindgen(start)]
pub fn start_app() {
    wasm_logger::init(wasm_logger::Config::default());
    yew::start_app::<App>();
}
