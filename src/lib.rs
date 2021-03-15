#![recursion_limit = "1024"]

use glenside::language::interpreter::Value;
use monaco::{
    api::CodeEditorOptions,
    sys::editor::BuiltinTheme,
    yew::{CodeEditor, CodeEditorLink},
};
use ndarray::Dimension;
use std::{collections::HashMap, rc::Rc};
use wasm_bindgen::prelude::*;
use yew::InputData;
use yew::{html, Component, ComponentLink, Html, ShouldRender};

fn get_options() -> CodeEditorOptions {
    CodeEditorOptions::default()
        .with_new_dimension(500, 500)
        .with_builtin_theme(BuiltinTheme::VsDark)
}

enum Message {
    NewInput,
}

struct App {
    options: Rc<CodeEditorOptions>,
    link: ComponentLink<Self>,
    code_editor_link: CodeEditorLink,
    result_text: String,
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
        }
    }

    fn update(&mut self, msg: Self::Message) -> ShouldRender {
        match msg {
            Message::NewInput => {
                let text_input = self
                    .code_editor_link
                    .with_editor(|editor| editor.get_model().unwrap().get_value())
                    .unwrap();

                let result = glenside::language::interpreter::interpret_from_str::<f64>(
                    &text_input,
                    &HashMap::new(),
                );

                let text_output = match result {
                    glenside::language::interpreter::Value::Tensor(t) => t.to_string(),
                    glenside::language::interpreter::Value::Access(_) => todo!(),
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
                <GeneratedTensorEnvironmentInput />
                <CodeEditor link=&self.code_editor_link, options=Rc::clone(&self.options) />
                <input type={"button"} value={"run"} onclick=self.link.callback(|_| Message::NewInput) />
                <input type={"text"} id={"output"} readonly={true} value={self.result_text.clone()} />
            </div>
        }
    }
}

trait EnvironmentInput<DataType> {
    fn get_value(&self) -> (String, Value<DataType>);
}

struct GeneratedTensorEnvironmentInput {
    link: ComponentLink<Self>,
    name: String,
    shape_string: String,
    value_generation_strategy: Option<ValueGenerationStrategy>,
}

impl EnvironmentInput<f64> for GeneratedTensorEnvironmentInput {
    fn get_value(&self) -> (String, Value<f64>) {
        let shape = self
            .shape_string
            .trim_start_matches(" ")
            .trim_start_matches("(")
            .trim_end_matches(" ")
            .trim_end_matches(")")
            .split(",")
            .map(|s| s.parse::<usize>().unwrap())
            .collect::<Vec<_>>();

        match self.value_generation_strategy {
            Some(ValueGenerationStrategy::Zeros) => (
                self.name.clone(),
                glenside::language::interpreter::Value::Tensor(ndarray::ArrayD::zeros(shape)),
            ),
            Some(ValueGenerationStrategy::Ones) => (
                self.name.clone(),
                glenside::language::interpreter::Value::Tensor(ndarray::ArrayD::ones(shape)),
            ),
            Some(ValueGenerationStrategy::Random) => todo!(),
            None => panic!(),
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
    type Properties = ();

    fn create(_props: Self::Properties, link: ComponentLink<Self>) -> Self {
        Self {
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
        false
    }

    fn change(&mut self, _props: Self::Properties) -> ShouldRender {
        todo!()
    }

    fn view(&self) -> Html {
        html! {
            <div>
                // Name text box
                <label for={"name"}>{"Name"}</label>
                <input name={"name"} type={"text"} oninput=self.link.callback(
                    |event: InputData| GeneratedTensorEnvironmentInputMessage::UpdateName(event.value)) />

                // Shape text box
                <label for={"shape"}>{"Shape (e.g. () or (3, 32, 32))"}</label>
                <input name={"shape"} type={"text"}
                    oninput=self.link.callback(|event: InputData| {
                        GeneratedTensorEnvironmentInputMessage::UpdateShapeString(event.value)
                    })
                />

                // Value generation radio buttons
                <input type={"radio"} id={"random"} name={"values"} value={"random"}
                    oninput=self.link.callback(|_|
                        GeneratedTensorEnvironmentInputMessage::UpdateValueGenerationStrategy(
                            ValueGenerationStrategy::Random
                        ))
                />
                <label for={"random"}>{"random"}</label>
                <input type={"radio"} id={"zeros"} name={"values"} value={"zeros"}
                    oninput=self.link.callback(|_|
                        GeneratedTensorEnvironmentInputMessage::UpdateValueGenerationStrategy(
                            ValueGenerationStrategy::Zeros
                        ))
                />
                <label for={"zeros"}>{"zeros"}</label>
                <input type={"radio"} id={"ones"} name={"values"} value={"ones"}
                    oninput=self.link.callback(|_|
                        GeneratedTensorEnvironmentInputMessage::UpdateValueGenerationStrategy(
                            ValueGenerationStrategy::Ones
                        ))
                />
                <label for={"ones"}>{"ones"}</label>
            </div>
        }
    }
}

#[wasm_bindgen(start)]
pub fn start_app() {
    yew::start_app::<App>();
}
