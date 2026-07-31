import { zhTwModule } from '../generated/zhTw.ts';
import { registerLanguageModule } from '../moduleRegistry.ts';

// Self-registers on import; the model is validated at registration.
registerLanguageModule(zhTwModule);

export { zhTwModule };
